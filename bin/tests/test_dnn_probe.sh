#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# probe fc3 to predict label (uv)
dnn_probe -anal uv -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -layer fc3 -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -beh label response -model svc -cv 2 -out $TMP_DIR

# probe fc3 to predict RT (mv)
dnn_probe -anal mv -act $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.act.h5 -layer fc3 -stim $DNNBRAIN_DATA/test/image/sub-CSI1_ses-01_imagenet.stim.csv -beh RT -model glm -cv 2 -out $TMP_DIR

