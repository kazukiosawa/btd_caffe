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
LOWRANK_DEPLOY=$MODEL_ROOT/lowrank/deploy.prototxt
LOWRANK_TRAIN_TEST=$MODEL_ROOT/lowrank/train_test.prototxt
LOWRANK_MODEL=$MODEL_ROOT/lowrank/VGG_ILSVRC_16_layers_lowrank.caffemodel
CONFIG=$MODEL_ROOT/lowrank/params.csv

# PCI setting
MAX_ITER=1000
MIN_DECREASE=1e-5

COMMAND=./approximate_net.py

$COMMAND $ORIGINAL_DEPLOY $ORIGINAL_MODEL \
         $TEMPLATE_DEPLOY $TEMPLATE_TRAIN_TEST \
         $LOWRANK_DEPLOY $LOWRANK_TRAIN_TEST $LOWRANK_MODEL $CONFIG \
         $MAX_ITER $MIN_DECREASE
