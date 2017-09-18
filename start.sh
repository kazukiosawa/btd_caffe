#!/bin/bash

# root setting
CAFFE_MODEL_ROOT=/opt/storage/models/caffe
MODEL_ROOT=$CAFFE_MODEL_ROOT/vgg16
PROTO_DIR=/workspace/btd_caffe/vgg16

# original model
ORIGINAL_DEPLOY=$PROTO_DIR/deploy.prototxt
ORIGINAL_TRAIN_TEST=$PROTO_DIR/train_test.prototxt
ORIGINAL_PARAMS=$MODEL_ROOT/VGG_ILSVRC_16_layers.caffemodel

# low-rank model
KEY=$1 
CONFIG=$PROTO_DIR/params_$KEY.csv
APPROX_DEPLOY=$PROTO_DIR/deploy_$KEY.prototxt
APPROX_TRAIN_TEST=$PROTO_DIR/train_test_$KEY.prototxt
APPROX_PARAMS=$MODEL_ROOT/VGG_ILSVRC_16_layers_$KEY.caffemodel

# PCI setting
MAX_ITER=100
MIN_DECREASE=1e-3

COMMAND=./approximate_net.py

DEPLOY="deploy"

if [ "$2" = $DEPLOY ]
 then
  $COMMAND --netdef $ORIGINAL_DEPLOY \
           --save_netdef $APPROX_DEPLOY \
	   --config $CONFIG 
else
  $COMMAND --netdef $ORIGINAL_TRAIN_TEST \
           --save_netdef $APPROX_TRAIN_TEST \
	   --config $CONFIG \
	   --params $ORIGINAL_PARAMS \
	   --save_params $APPROX_PARAMS \
           --max_iter $MAX_ITER \
	   --min_decrease $MIN_DECREASE 
fi
