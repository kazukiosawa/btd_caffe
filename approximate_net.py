#!/usr/bin/python

import sys
import numpy as np
import csv
import caffe
from caffe.proto import caffe_pb2
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
			param, c, n, blocks = row[0], int(row[1]), int(row[2]), int(row[3])
			btd_config[param] = (c, n, blocks)
	return btd_config


def create_new_net(template_deploy, template_train_test, save_deploy, save_train_test, btd_config):
	new_deploy, new_train_test = \
		get_template_net(args.template_deploy, args.template_train_test)
	modify_convolution_num_output(new_deploy, btd_config)
	modify_convolution_num_output(new_train_test, btd_config)
	# save new prototxt
	with open(save_deploy, 'w') as f:
		f.write(str(new_deploy))
	with open(save_train_test, 'w') as f:
		f.write(str(new_train_test))
	# load new prototxt
	save_net = caffe.Net(save_deploy, caffe.TEST)
	return save_net


def get_template_net(template_deploy, template_train_test):
	# create new prototxt
	new_deploy = caffe_pb2.NetParameter()
	new_train_test = caffe_pb2.NetParameter()

	# load template prototxt
	with open(template_deploy) as f:
		s = f.read()
		txtf.Merge(s, new_deploy)
	with open(template_train_test) as f:
		s = f.read()
		txtf.Merge(s, new_train_test)
	return new_deploy, new_train_test


def modify_convolution_num_output(new_proto, btd_config):
	layer_names = [l.name for l in new_proto.layer]
	for conv, values in btd_config.items():
		c, n, blocks = values
		idx = layer_names.index(conv + 'a')
		l = new_proto.layer[idx]
		l.convolution_param.num_output = c
		idx = layer_names.index(conv + 'b')
		l = new_proto.layer[idx]
		l.convolution_param.num_output = n
		l.convolution_param.group = blocks


# approximate original weights (kernels) & copy original bias
def create_lowrank_net(net_lowrank, net, btd_config, max_iter, min_decrease):
	copy = [param for param in net.params.keys() if not param in btd_config.keys()]
	for param in copy:
		net_lowrank.params[param][0].data[...] = net.params[param][0].data
		net_lowrank.params[param][1].data[...] = net.params[param][1].data

	convs = [(k, v[0].data, v[1].data) for k, v in net.params.items() if k in btd_config.keys()]
	for conv, kernel, bias in convs:
		size = kernel.shape
		N, C, H, W = size[0:4]
		P = H * W
		c, n, blocks = btd_config[conv]
		# (N, C, H, W) -> (N, C, P)
	        kernel = kernel.reshape(N, C, P)	
		# compute BTD 
		n_ = int(n/blocks)
		c_ = int(c/blocks)
		rank = [n_, c_, P]
		print ('calculating BTD for {0}...').format(conv)
		btd, _ = pci(dtensor(kernel), blocks, rank, max_iter, min_decrease) 
		# BTD -> (c, C) (n, c/blocks, P) (N, n)
		kernel_a = np.concatenate([subtensor[1][1] for subtensor in btd], axis=1)
		kernel_b = np.concatenate([subtensor[0].ttm(subtensor[1][2], 2) for subtensor in btd], axis=0)
		kernel_c = np.concatenate([subtensor[1][0] for subtensor in btd], axis=1)
		# (c, C) -> (c, C, 1, 1)
		kernel_a = kernel_a.T.reshape(c, C, 1, 1)
		# (n, c/blocks, P) -> (n, c/blocks, H, W)
		kernel_b = kernel_b.reshape(n, c_, H, W)
		# (N, n) -> (N, n, 1, 1)
		kernel_c = kernel_c.reshape(N, n, 1, 1)
		# set kernel to low-rank model
		net_lowrank.params[conv + 'a'][0].data[...] = kernel_a
		net_lowrank.params[conv + 'b'][0].data[...] = kernel_b
		net_lowrank.params[conv + 'c'][0].data[...] = kernel_c
		# copy bias to low-rank model
		net_lowrank.params[conv + 'c'][1].data[...] = bias
	return net_lowrank	


def main(args):
	if args.max_iter is None:
		max_iter=100
	else:
		max_iter=int(args.max_iter)
	if args.min_decrease is None:
		min_decrease=1e-5
	else:
		min_decrease=float(args.min_decrease)

	btd_config = load_config(args.config)
	new_net = create_new_net(args.template_deploy, args.template_train_test \
				, args.save_deploy, args.save_train_test, btd_config)
	# load original model
	net = caffe.Net(args.model, args.weights, caffe.TEST)
	net_lowrank = create_lowrank_net(new_net, net, btd_config, max_iter, min_decrease)
	# save low-rank model
	print ('saving {0} ...').format(args.save_weights)
	net_lowrank.save(args.save_weights)
	print ('done.')


if __name__ == '__main__':
    parser = ArgumentParser(description="Block Term Decomposition on Convolution Kernel")
    parser.add_argument('--model', required=True,
        help="Prototxt of the original net")
    parser.add_argument('--weights', required=True,
        help="Caffemodel of the original net")
    parser.add_argument('--save_deploy', required=True,
        help="Path to the deploy.prototxt of the low-rank approximated net")
    parser.add_argument('--save_train_test', required=True,
        help="Path to the deploy.prototxt of the low-rank approximated net")
    parser.add_argument('--save_weights', required=True,
        help="Path to the caffemodel of the low-rank approximated net")
    parser.add_argument('--config', required=True,
        help="CSV config file for BTD")
    parser.add_argument('--max_iter',
        help="Max iteration for BTD")
    parser.add_argument('--min_decrease',
        help="Minimum error decrease in each iteration for BTD")
    parser.add_argument('--template_deploy', required=True,
        help="Path to the deploy.prototxt of the template net")
    parser.add_argument('--template_train_test', required=True,
        help="Path to the train_test.prototxt of the template net")
    args = parser.parse_args()
    main(args)
