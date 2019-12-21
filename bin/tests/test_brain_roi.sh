#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

brain_roi -nif $DNNBRAIN_DATA/test/sub-CSI1_ses-01_imagenet_beta_L.nii.gz -mask $DNNBRAIN_DATA/test/MMP_L.nii.gz -out $TMP_DIR/test.roi.h5
