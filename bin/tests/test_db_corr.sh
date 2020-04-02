#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# .nii
db_corr -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -dmask $DNNBRAIN_DATA/test/alexnet.dmask.csv -iteraxis row_col -resp $DNNBRAIN_DATA/test/sub-CSI1_ses-01_imagenet_beta_L.nii.gz -bmask $DNNBRAIN_DATA/test/PHA1_L.nii.gz -out $TMP_DIR

# .roi.h5
db_corr -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -dmask $DNNBRAIN_DATA/test/alexnet.dmask.csv -iteraxis channel -resp $DNNBRAIN_DATA/test/PHA1.roi.h5 -roi PHA1_R -out $TMP_DIR
