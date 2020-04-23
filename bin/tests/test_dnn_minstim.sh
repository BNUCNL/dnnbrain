#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# test
dnn_minstim -net AlexNet -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -layer conv5 -chn 1 -out $TMP_DIR/min_stim

