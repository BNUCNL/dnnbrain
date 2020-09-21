Minimal Image
=============

Minimal image is a way to simplify a stimulus into a minimal part which could cause
almost equivalent activation as the original stimlus using DNN, which you can see the 
concept in the passage by `Ullman et al. <https://www.pnas.org/content/113/10/2744>`__
and by `Srivastava et al. <https://arxiv.org/abs/1902.03227>`__

Before generating the minimal image, you need to ensure that the images can be highly activated by the given unit in DNN,
which can be obtained using `dnn_topstim <https://dnnbrain.readthedocs.io/en/latest/docs/cmd/dnn_topstim.html>`__
(Select the topK stimuli from a stimulus set).

In DNNBrain, we provide another novel way to generate minimal image. First, we decompose a image into multiple parcels
using methods by `skimage. <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html#sphx-glr-auto-examples-segmentation-plot-segmentations-py>`__
Then we sort these parcels based on their activations by the given unit in DNN and 
combine them iterally. The combined-parcel image will be compared its activation with the original image and finally 
get the minimal image.

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
   from dnnbrain.dnn.algo import MinimalParcelImage

   # Prepare DNN and image
   dnn = AlexNet()
   image = plt.imread('ILSVRC_val_00095233.JPEG')

   # Generate minimal image using felzenszwalb method
   # which cause almost equivalent activation as  
   # the raw stimlus' activation of the 122th unit of conv5.
   img_min = MinimalParcelImage(dnn, 'conv5', 122)
   img_min.felzenszwalb_decompose(image) 
   img_out = img_min.generate_minimal_image()

   # transform to PIL image and save out
   img_out = ip.to_pil(img_out.transpose(2,0,1), True)
   img_out.save('ILSVRC_val_00095233.JPEG_min.JPEG')

The minimal image is displayed as below:

.. raw:: html

   <center>

|minimal|

.. raw:: html

   </center>

.. |original| image:: ../img/ILSVRC_val_00095233.JPEG
.. |minimal| image:: ../img/ILSVRC_val_00095233_min.JPEG

