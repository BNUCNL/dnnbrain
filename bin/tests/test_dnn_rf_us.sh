#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# test
dnn_rf_us -net AlexNet -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -layer conv5 -channel 1 -ip_metric bicubic -up_thres 0.68 -out $TMP_DIR/rf_us

