#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# saliency image via Vanilla
dnn_saliency -net AlexNet -layer fc3 -chn 294 23 -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -show -out $TMP_DIR -meth vanilla

# saliency image via Guided
dnn_saliency -net AlexNet -layer fc3 -chn 294 23 -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -show -out $TMP_DIR
