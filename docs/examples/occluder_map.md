# Receptive Field Occulding
There is an example of visualizing receptive field using slide-window(sw) occulding method  through using python library of DNNBrain.

The original image used in this doc is displayed as below:
<center>![original](../../img/ILSVRC_val_00095233.JPEG)</center>
## Example 
```
import numpy as np
import matplotlib.pyplot as plt

from dnnbrain.dnn.base import ip
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.algo import OccluderDiscrepancyMapping

# Prepare DNN and image
dnn = AlexNet()
image = plt.imread('ILSVRC_val_00095233.JPEG')

# Visualizing receptive field using slide-window(sw) occulding method
# which displays the receptive field that contribute to 
# the activation of the 122th unit of conv5.
oc_estimator = OccluderDiscrepancyMapping(dnn, 'conv5', 122)
oc_estimator.set_params(window=(11,11), stride=(2,2), metric='max')
img_out = oc_estimator.compute(image)

# transform to PIL image and save out
img_out = ip.to_pil(img_out, True)
img_out.save('ILSVRC_val_00095233_rf_sw.JPEG')
```
The receptive field occulding image is displayed as below:
<center>![occluding](../../img/ILSVRC_val_00095233_rf_sw.JPEG)</center>
