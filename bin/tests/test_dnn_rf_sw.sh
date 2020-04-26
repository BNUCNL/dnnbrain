#! /bin/bash

TMP_DIR=~/.dnnbrain_tmp
mkdir -p $TMP_DIR

# test
dnn_rf_sw -net AlexNet -layer conv5 -chn 125 -stim /nfs/s2/userhome/zhouming/workingdir/stim/test_rf.stim.csv -wsize 11 11 -stride 2 2 -metric max -out $TMP_DIR/rf_sw

