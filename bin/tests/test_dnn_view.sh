#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# extract DNN activation from video with -layer
dnn_view -net AlexNet -layer conv5 -chn 60 125 -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -show -out $TMP_DIR/
