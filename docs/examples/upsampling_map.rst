Receptive Field Upsampling
==========================

There is an example of visualizing receptive field using up-sampling(us)
method through using python library of DNNBrain.

The original image used in this doc is displayed as below:

.. raw:: html

   <center>

|original|

.. raw:: html

   </center>

Example
-------

::

   import numpy as np
   import matplotlib.pyplot as plt

   from dnnbrain.dnn.base import ip
   from dnnbrain.dnn.models import AlexNet
   from dnnbrain.dnn.algo import UpsamplingActivationMapping

   # Prepare DNN and image
   dnn = AlexNet()
   image = plt.imread('ILSVRC_val_00095233.JPEG')

   # Visualizing receptive field using up-sampling(us) method
   # which displays the receptive field that contribute to 
   # the activation of the 122th unit of conv5.
   up_estimator =UpsamplingActivationMapping(dnn, 'conv5', 122)
   up_estimator.set_params(interp_meth='bicubic', interp_threshold=0.95)
   img_out = up_estimator.compute(image)

   # transform to PIL image and save out
   img_out = ip.to_pil(img_out, True)
   img_out.save('ILSVRC_val_00095233_rf_us.JPEG')

The receptive field upsampling is displayed as below:

.. raw:: html

   <center>

|vanilla|

.. raw:: html

   </center>

.. |original| image:: ../img/ILSVRC_val_00095233.JPEG
.. |vanilla| image:: ../img/ILSVRC_val_00095233_rf_us.JPEG
