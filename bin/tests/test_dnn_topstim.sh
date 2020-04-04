#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# one layer
dnn_topstim -net AlexNet -top 3 -layer fc3 -chn 1 2 -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -out $TMP_DIR/top_stim1

# two layer
dnn_topstim -net AlexNet -top 3 -layer conv5 fc3 -chn 1 2 -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -out $TMP_DIR/top_stim2
