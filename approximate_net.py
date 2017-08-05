#!/usr/bin/python

import numpy as np
import csv
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
from pci import pci
from sktensor import dtensor

caffe_model_root = '/opt/storage/models/caffe'
model_root = caffe_model_root + '/vgg16'

original_deploy = model_root + '/deploy.prototxt'
original_model = model_root + '/VGG_ILSVRC_16_layers.caffemodel'
template_deploy = model_root + '/lowrank/template_deploy.prototxt'
template_train_test = model_root + '/lowrank/template_train_test.prototxt'
lowrank_deploy = model_root + '/lowrank/deploy.prototxt'
lowrank_train_test = model_root + '/lowrank/train_test.prototxt'
lowrank_model = model_root + '/lowrank/VGG_ILSVRC_16_layers_lowrank.caffemodel'
params_def = model_root + '/lowrank/params.csv'

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

# load ranks for low-rank approximation
with open(params_def, 'r') as f:
	reader = csv.reader(f)
	params = {row[0]:(int(row[1]), int(row[2]), int(row[3])) for row in reader}

def modify_convolution_num_output(new_proto, params):
	layer_names = [l.name for l in new_proto.layer]
	for conv, values in params.items():
		c, n, _ = values
		idx = layer_names.index(conv + 'a')
		l = new_proto.layer[idx]
		l.convolution_param.num_output = c
		idx = layer_names.index(conv + 'b')
		l = new_proto.layer[idx]
		l.convolution_param.num_output = n

modify_convolution_num_output(new_deploy, params)
modify_convolution_num_output(new_train_test, params)

# save new prototxt
with open(lowrank_deploy, 'w') as f:
	f.write(str(new_deploy))
with open(lowrank_train_test, 'w') as f:
	f.write(str(new_train_test))

# load new prototxt
net_lowrank = caffe.Net(lowrank_deploy, caffe.TEST)

# load original model
net = caffe.Net(original_deploy, original_model, caffe.TEST)

# load original params
exclude = ['conv1_1']
for name in exclude:
	net_lowrank.params[name][0].data[...] = net.params[name][0].data

# approximate original params
convs = [(k, v[0].data) for k, v in net.params.items() if 'conv' in k and not k in exclude]
for conv, kernel in convs:
	size = kernel.shape
	N,C,H,W = size[0:4]
	P = H * W
	c, n, blocks = params[conv]
	# (N, C, H, W) -> (N, C, P)
        kernel = kernel.reshape(N, C, P)	
	# compute BTD 
	n_ = int(n/blocks)
	c_ = int(c/blocks)
	rank = [n_, c_, P]
	print ('calculating BTD for {0}...').format(conv)
        btd, _ = pci(dtensor(kernel), blocks, rank) 
	print ('finished.')
	# BTD -> (c, C) (n, c, P) (N, n)
	kernel_a = np.c_[[subtensor[1][1] for subtensor in btd]]
	kernel_b = np.zeros((n, c, P))
	for (i, subtensor) in enumerate(btd):
		core = subtensor[0]
		kernel_b[i*n_:(i+1)*n_, i*c_:(i+1)*c_, :] = core
	kernel_c = np.r_[[subtensor[1][0] for subtensor in btd]]
	# (c, C) -> (c, C, 1, 1)
	kernel_a = kernel_a.reshape(c, C, 1, 1)
	# (n, c, P) -> (n, c, H, W)
	kernel_b = kernel_b.reshape(n, c, H, W)
	# (N, n) -> (N, n, 1, 1)
	kernel_c = kernel_c.reshape(N, n, 1, 1)
	# set kernel to low-rank model
	net_lowrank.params[conv + 'a'][0].data[...] = kernel_a
	net_lowrank.params[conv + 'b'][0].data[...] = kernel_b
	net_lowrank.params[conv + 'c'][0].data[...] = kernel_c

# save caffemodel of low-rank model
net_lowrank.save(lowrank_model)
