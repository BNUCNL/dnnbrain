#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# extract DNN activation from images
dnn_act -net alexnet -layer conv5 fc3 -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -out $TMP_DIR/dnn_act_image.act.h5

# extract DNN activation from video
dnn_act -net alexnet -layer conv5 fc3 -stim $DNNBRAIN_DATA/test/video/sub-CSI1_ses-01_imagenet.stim.csv -out $TMP_DIR/dnn_act_video.act.h5
