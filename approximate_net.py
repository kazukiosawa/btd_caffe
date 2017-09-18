#!/usr/bin/python

import sys
import numpy as np
import csv
import caffe
from caffe import layers as L
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter
import google.protobuf.text_format as txtf
from pci import pci
from sktensor import dtensor
from argparse import ArgumentParser


# load parameters for btd
def load_config(config_file):
	btd_config = {}
	with open(config_file, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			param, s, t, blocks = row[0], int(row[1]), int(row[2]), int(row[3])
			btd_config[param] = (s, t, blocks)
	return btd_config


def create_approx_netdef(input_file, output_file, btd_config):
	with open(input_file, 'r') as fp:
		net = NetParameter()
		txtf.Merge(fp.read(), net)
	new_layers = []
	for layer in net.layer:	
		if not layer.name in btd_config.keys():
			new_layers.append(layer)
			continue
		s, t, r = btd_config[layer.name]
		a, b, c = decompose2abc(layer, s, t, r)
		new_layers.extend([a, b, c])
	new_net = NetParameter()
	new_net.CopyFrom(net)
	del(new_net.layer[:])
	new_net.layer.extend(new_layers)
	with open(output_file, 'w') as fp:
		fp.write(txtf.MessageToString(new_net))
		

def decompose2abc(conv, s, t, r):
	print conv.name
	def _create_new(name):
		print "--> {0}".format(name)
		new_ = LayerParameter()
		new_.CopyFrom(conv)
		new_.name = name
		return new_
	conv_param = conv.convolution_param
	# 1st
	a = _create_new(conv.name + 'a')
	del(a.top[:])
	a.top.extend([a.name])
	if len(a.param) > 2:
		a.param[1].lr_mult = 0
	a_param = a.convolution_param
	a_param.num_output = s
	del(a_param.kernel_size[:])
	a_param.kernel_size.extend([1])
	del(a_param.pad[:])
	a_param.pad.extend([0])
	del(a_param.stride[:])
	a_param.stride.extend([1])
	# 2nd
	b = _create_new(conv.name + 'b')
	del(b.bottom[:])
	b.bottom.extend(a.top)
	del(b.top[:])
	b.top.extend([b.name])
	if len(b.param) > 2:
		b.param[1].lr_mult = 0
	b_param = b.convolution_param
	b_param.num_output = t
	b_param.group = r
	# 3rd
	c = _create_new(conv.name + 'c')
	del(c.bottom[:])
	c.bottom.extend(b.top)
	c_param = c.convolution_param
	del(c_param.kernel_size[:])
	c_param.kernel_size.extend([1])
	del(c_param.pad[:])
	c_param.pad.extend([0])
	del(c_param.stride[:])
	c_param.stride.extend([1])
	return a, b, c


def approximate_params(netdef, params, approx_netdef, approx_params,
			 btd_config, max_iter, min_decrease):
	net = caffe.Net(netdef, params, caffe.TEST)
	net_approx = caffe.Net(approx_netdef, params, caffe.TEST)
	convs = [(k, v[0].data, v[1].data) for k, v in net.params.items() if k in btd_config.keys()]
	for conv, kernel, bias in convs:
		size = kernel.shape
		T, S, H, W = size[0:4]
		P = H * W
		s, t, blocks = btd_config[conv]
		# (T, S, H, W) -> (T, S, P)
	        kernel = kernel.reshape(T, S, P)	
		# compute BTD 
		t_ = int(t/blocks)
		s_ = int(s/blocks)
		rank = [t_, s_, P]
		print ('calculating BTD for {0}...').format(conv)
		btd, _ = pci(dtensor(kernel), blocks, rank, max_iter, min_decrease) 
		# BTD -> (c, C) (n, c/blocks, P) (N, n)
		kernel_a = np.concatenate([subtensor[1][1] for subtensor in btd], axis=1)
		kernel_b = np.concatenate([subtensor[0].ttm(subtensor[1][2], 2) for subtensor in btd], axis=0)
		kernel_c = np.concatenate([subtensor[1][0] for subtensor in btd], axis=1)
		# (c, C) -> (c, C, 1, 1)
		kernel_a = kernel_a.T.reshape(s, S, 1, 1)
		# (n, c/blocks, P) -> (n, c/blocks, H, W)
		kernel_b = kernel_b.reshape(t, s_, H, W)
		# (N, n) -> (N, n, 1, 1)
		kernel_c = kernel_c.reshape(T, t, 1, 1)
		# set kernel to low-rank model
		net_approx.params[conv + 'a'][0].data[...] = kernel_a
		net_approx.params[conv + 'b'][0].data[...] = kernel_b
		net_approx.params[conv + 'c'][0].data[...] = kernel_c
		# copy bias to low-rank model
		net_approx.params[conv + 'c'][1].data[...] = bias
	net_approx.save(approx_params)


def main(args):
	btd_config = load_config(args.config)
	create_approx_netdef(args.netdef, args.save_netdef, btd_config)

	if args.params is None: return
	if args.max_iter is None:
		max_iter=100
	else:
		max_iter=int(args.max_iter)
	if args.min_decrease is None:
		min_decrease=1e-5
	else:
		min_decrease=float(args.min_decrease)

	approximate_params(args.netdef, args.params, args.save_netdef, args.save_params, 
				btd_config, max_iter, min_decrease)


if __name__ == '__main__':
    parser = ArgumentParser(description="Block Term Decomposition on Convolution Kernel")
    parser.add_argument('--netdef', required=True,
        help="Prototxt of the original net")
    parser.add_argument('--save_netdef', required=True,
        help="Path to the deploy.prototxt of the low-rank approximated net")
    parser.add_argument('--config', required=True,
        help="CSV config file for BTD")
    parser.add_argument('--params', 
        help="Caffemodel of the original net")
    parser.add_argument('--save_params',
        help="Path to the caffemodel of the low-rank approximated net")
    parser.add_argument('--max_iter',
        help="Max iteration for BTD")
    parser.add_argument('--min_decrease',
        help="Minimum error decrease in each iteration for BTD")
    args = parser.parse_args()
    main(args)
