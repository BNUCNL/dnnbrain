#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

dnn_fe -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -layer conv5 fc3 -meth pca 3 -axis chn -out $TMP_DIR/dnn_fe.act.h5
