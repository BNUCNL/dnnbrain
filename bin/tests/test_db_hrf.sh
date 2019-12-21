#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

db_hrf -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -layer fc3 -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -tr 2 -n_vol 10 -out $TMP_DIR/db_hrf.act.h5
