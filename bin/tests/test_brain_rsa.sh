#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

brain_rsa -nif $DNNBRAIN_DATA/test/sub-CSI1_ses-01_imagenet_beta_L.nii.gz -bmask $DNNBRAIN_DATA/test/PHA1_L.nii.gz -roi 1 -out $TMP_DIR/brain_rsa.rdm.h5

brain_rsa -nif $DNNBRAIN_DATA/test/sub-CSI1_ses-01_imagenet_beta_L.nii.gz -bmask $DNNBRAIN_DATA/test/PHA1_L.nii.gz -roi 1 -cate $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -out $TMP_DIR/brain_rsa_cate.rdm.h5
