#!/bin/bash

# root setting
CAFFE_MODEL_ROOT=/opt/storage/models/caffe
MODEL_ROOT=$CAFFE_MODEL_ROOT/vgg16

# original model
ORIGINAL_DEPLOY=$MODEL_ROOT/deploy.prototxt
ORIGINAL_MODEL=$MODEL_ROOT/VGG_ILSVRC_16_layers.caffemodel

# template for low-rank model
TEMPLATE_DEPLOY=$MODEL_ROOT/lowrank/template_deploy.prototxt
TEMPLATE_TRAIN_TEST=$MODEL_ROOT/lowrank/template_train_test.prototxt

# low-rank model 
LOWRANK_DEPLOY=$MODEL_ROOT/lowrank/deploy2.prototxt
LOWRANK_TRAIN_TEST=$MODEL_ROOT/lowrank/train_test2.prototxt
LOWRANK_MODEL=$MODEL_ROOT/lowrank/VGG_ILSVRC_16_layers_lowrank2.caffemodel
CONFIG=$MODEL_ROOT/lowrank/params2.csv

# PCI setting
MAX_ITER=1000
MIN_DECREASE=1e-2

COMMAND=./approximate_net.py

$COMMAND --model $ORIGINAL_DEPLOY \
	 --weights $ORIGINAL_MODEL \
         --save_deploy $LOWRANK_DEPLOY \
	 --save_train_test $LOWRANK_TRAIN_TEST \
	 --save_weights $LOWRANK_MODEL \
	 --config $CONFIG \
         --max_iter $MAX_ITER \
	 --min_decrease $MIN_DECREASE \
         --template_deploy $TEMPLATE_DEPLOY \
	 --template_train_test $TEMPLATE_TRAIN_TEST
