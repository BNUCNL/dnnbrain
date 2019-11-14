#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# do uva with .nii
db_decode -anal uva -resp $DNNBRAIN_DATA/test/sub-CSI1_ses-01_imagenet_beta_L.nii.gz -bmask $DNNBRAIN_DATA/test/PHA1_L.nii.gz -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -dmask $DNNBRAIN_DATA/test/alexnet.dmask.csv -model glm -out $TMP_DIR

# do mva with .nii
db_decode -anal mva -resp $DNNBRAIN_DATA/test/sub-CSI1_ses-01_imagenet_beta_L.nii.gz -bmask $DNNBRAIN_DATA/test/PHA1_L.nii.gz -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -dmask $DNNBRAIN_DATA/test/alexnet.dmask.csv -model glm -out $TMP_DIR

# do uva with .roi.h5
db_decode -anal uva -resp $DNNBRAIN_DATA/test/PHA1.roi.h5 -roi PHA1_R -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -dmask $DNNBRAIN_DATA/test/alexnet.dmask.csv -model glm -out $TMP_DIR

# do mva with .roi.h5
db_decode -anal mva -resp $DNNBRAIN_DATA/test/PHA1.roi.h5 -roi PHA1_R -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -dmask $DNNBRAIN_DATA/test/alexnet.dmask.csv -model glm -out $TMP_DIR

