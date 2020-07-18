Empirical Map
=========================

Empirical map is a way to visualize a set of images' receptive field. 

Before generating the empirical map, you need to ensure that the images can be highly activated by the given unit in DNN,
which can be obtained using `dnn_topstim <https://dnnbrain.readthedocs.io/en/latest/docs/cmd/dnn_topstim.html>`__
(Select the topK stimuli from a stimulus set).

When generating the empirical map, you need to first assign an engine to get regions of the image 
lead to the high unit activations(occluding or upsampling). After acquiring these regions, 
the patch will be averaged and get the final empirical map.

There is an example of empirical map using up-sampling(us) method 
through using python library of DNNBrain.

The original images used in this doc are displayed as below:

.. raw:: html

   <center>

|original|

.. raw:: html

   </center>

Example
-------

::

   from dnnbrain.dnn.base import ip
   from dnnbrain.dnn.core import Stimulus
   from dnnbrain.dnn.models import AlexNet
   from dnnbrain.dnn.algo import EmpiricalReceptiveField, UpsamplingActivationMapping

   # Prepare DNN and stimulus
   dnn = AlexNet()
   stim = Stimulus()
   stim.load('test.stim.csv')

   # Visualizing empirical receptive field using up-sampling(us) engine
   # which displays the receptive field that contribute to 
   # the activation of the 122th unit of conv5.
   up_estimator =UpsamplingActivationMapping(dnn, 'conv5', 122)
   up_estimator.set_params(interp_meth='bicubic', interp_threshold=0.50)
   emp_rf = EmpiricalReceptiveField(up_estimator)
   img_out = emp_rf.compute(stim)

   # transform to PIL image and save out
   img_out = ip.to_pil(img_out, True)
   img_out.save('empirical_rf.JPEG')

The empirical receptive field image is displayed as below:

.. raw:: html

   <center>

|emprical|

.. raw:: html

   </center>

.. |original| image:: ../img/empirical_org.JPEG
.. |emprical| image:: ../img/empirical_rf.JPEG

