#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

dnn_pooling -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -meth max -out $TMP_DIR/dnn_pooling.act.h5
