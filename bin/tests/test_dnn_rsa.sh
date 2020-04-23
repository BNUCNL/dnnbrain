#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

dnn_rsa -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -dmask $DNNBRAIN_DATA/test/alexnet.dmask.csv -iteraxis channel -metric correlation -out $TMP_DIR/dnn_rsa_dmask.rdm.h5

dnn_rsa -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -layer conv5 fc3 -metric euclidean -out $TMP_DIR/dnn_rsa_layer.rdm.h5

dnn_rsa -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -out $TMP_DIR/dnn_rsa.rdm.h5
