# Extract DNN Activation
There are some examples of DNN activation extraction through using python library of DNNBrain.

## Example 1
Extracting activation of stimuli loaded from files.
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
## Example 2
Extracting activation of stimuli that is already a numpy array.
```
import os
import numpy as np

from os.path import join as pjoin
from dnnbrain.dnn.core import Mask
from dnnbrain.dnn.models import AlexNet

TMP_DIR = pjoin(os.environ['HOME'], '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

# The stimulus array's shape must be (n_stim, n_chn, height, width).
# Here we make up a stimulus array with shape (2, 3, 224, 224).
# It represent 2 RGB images with size (224, 224).
# Note, all elements in stimulus array must be contained in [0, 255].
# And their data type must be 8-bit unsigned integer.
stimuli = np.random.randint(0, 256, (2, 3, 224, 224), dtype=np.uint8)

# Set DNN mask
# As a result, we will extract activation of 
# the 1st and 3rd channels of layer conv5 and layer fc3.
dmask = Mask()
dmask.set('conv5', channels=[1, 3])
dmask.set('fc3')

# Extract DNN activation
dnn = AlexNet()
activation = dnn.compute_activation(stimuli, dmask)

# Save out
out_file = pjoin(TMP_DIR, 'extract.act.h5')
activation.save(out_file)
```
