Layer to Module
===============

DNNBrain allows users to conveniently approach to the module inside a
built-in DNN by a commonly used layer name. For each built-in DNN, its
available layer names and the mapping between layer names and modules
are shown as follow. And we take the AlexNet as example to interpret
mechanism of the mapping.

AlexNet
-------

Generally, we can approach to a module by its location in the DNN
framework. For example, according to AlexNet framework (Fig. 1, left),
we can locate the first convolutional layer by its location (features,
0). However, DNNBrain users can easily do it by the layer name ‘conv1’
according to the mapping (Fig. 1, right).

.. raw:: html

   <center>
   
|AlexNet_framework|
|AlexNet_mapping|

Figure 1

.. raw:: html

   </center>

VggFace
-------

.. raw:: html

   <center>

|VggFace_framework|
|VggFace_mapping|

Figure 2

.. raw:: html

   </center>

Vgg11
-----

.. raw:: html

   <center>

|Vgg11_framework|
|Vgg11_mapping|

Figure 3

.. raw:: html

   </center>


.. |AlexNet_framework| image:: ../img/layer2module/alexnet_framework.png
   :width: 64%
.. |AlexNet_mapping| image:: ../img/layer2module/alexnet_mapping.png
   :width: 35%
.. |VggFace_framework| image:: ../img/layer2module/vggface_framework.png
   :width: 64%
.. |VggFace_mapping| image:: ../img/layer2module/vggface_mapping.png
   :width: 35%
.. |Vgg11_framework| image:: ../img/layer2module/vgg11_framework.png
   :width: 64%
.. |Vgg11_mapping| image:: ../img/layer2module/vgg11_mapping.png
   :width: 35%
