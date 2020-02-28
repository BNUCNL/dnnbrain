# Extract DNN Activation
&emsp;&emsp;This is an example of DNN activation extraction through using python library of DNNBrain.
```
import os
import numpy as np

from os.path import join as pjoin
from dnnbrain.dnn.core import Stimulus, Mask
from dnnbrain.dnn.models import AlexNet

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.environ['HOME'], '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

# Load stimuli information
stim_file = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.stim.csv')
stimuli = Stimulus()
stimuli.load(stim_file)

# Load mask information
dmask_file = pjoin(DNNBRAIN_TEST, 'alexnet.dmask.csv')
dmask = Mask()
dmask.load(dmask_file)

# Extract DNN activation
dnn = AlexNet()
activation = dnn.compute_activation(stimuli, dmask)

# Save out
out_file = pjoin(TMP_DIR, 'extract.act.h5')
activation.save(out_file)
```