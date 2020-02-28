#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# without .dmask.csv
dnn_mask -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -layer conv5 -chn 1 2 3 -out $TMP_DIR/dnn_mask1.act.h5

# use .dmask.csv
dnn_mask -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -dmask $DNNBRAIN_DATA/test/alexnet.dmask.csv -out $TMP_DIR/dnn_mask2.act.h5
