Name
====

dnn_minstim - Simplify a stimulus into a minimal part which could cause
almost equivalent activation as the original stimlus using DNN

Synopsis
========

dnn_minstim -net Net -layer Layer -chn Channel -stim Stimulus -out
Output

Arguments
=========

Required Arguments
------------------

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| net                         | Name of a neural network               |
+-----------------------------+----------------------------------------+
| stim                        | Path of a .stim.csv file which         |
|                             | contains stimulus information          |
+-----------------------------+----------------------------------------+
| layer                       | Name of the target layer.Only support  |
|                             | one layer each time.E.g., ‘conv1’      |
|                             | represents the first convolution       |
|                             | layer, and ‘fc1’ represents the first  |
|                             | full connection layer.                 |
+-----------------------------+----------------------------------------+
| chn                         | Index of target channel.Only support   |
|                             | one channel each time.Channel index    |
|                             | starts from 1.                         |
+-----------------------------+----------------------------------------+
| out                         | Output path to save the simplfied      |
|                             | image                                  |
+-----------------------------+----------------------------------------+

Outputs
=======

A series of minimal images based on your stimilus and interested net
information

Examples
========

::

   dnn_minstim -net AlexNet -layer conv5 -chn 122 -stim ./flowers.stim.csv -out ./image/minstim/

The original image used in this doc is displayed as below:

.. raw:: html

   <center>

|original|

.. raw:: html

   </center>

The minimal image is displayed as below:

.. raw:: html

   <center>

|minimal|

.. raw:: html

   </center>

.. |original| image:: ../../img/ILSVRC_val_00095233.JPEG
.. |minimal| image:: ../../img/ILSVRC_val_00095233_min.JPEG

