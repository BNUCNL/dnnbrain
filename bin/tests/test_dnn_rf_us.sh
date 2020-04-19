#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# test
dnn_rf_us -net AlexNet -stim /nfs/s2/userhome/zhouming/workingdir/stim/test_rf.stim.csv -layer conv5 -chn 125 -ip_metric bicubic -up_thres 0.68 -out $TMP_DIR/rf_us

