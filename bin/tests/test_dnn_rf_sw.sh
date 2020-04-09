#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# test
dnn_rf_sw -net AlexNet -layer conv3 -channel 1 -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -out $TMP_DIR/rf_sw

