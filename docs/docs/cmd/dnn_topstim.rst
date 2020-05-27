Name
====

dnn_topstim - For maximizing activation of a specific layer and channel
in DNN network, selecting the topK stimuli from a stimulus set.
Meanwhile, save their activation.

Synopsis
========

::

   dnn_topstim [-h] -net Net -top TopNumber -stim Stimulus -layer Layer
               [Layer ...] -chn Channel [Channel ...] [-cuda] -out OutputDir

Arguments
=========

Required Arguments
------------------

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| net                         | Name of DNN Model, which should be     |
|                             | placed in $DNNBRAIN_DATA/models with   |
|                             | format \*.pth.                         |
+-----------------------------+----------------------------------------+
| top                         | Number of top stimulus. For example,   |
|                             | assign top=5, and top 5 image for each |
|                             | <layer,channel> pair will be selected. |
+-----------------------------+----------------------------------------+
| stim                        | a .stim.csv file which contains        |
|                             | stimulus information                   |
+-----------------------------+----------------------------------------+
| layer                       | names of the layers used to specify    |
|                             | where activation is extracted fromFor  |
|                             | example, ‘conv1’ represents the first  |
|                             | convolution layer, and ‘fc1’           |
|                             | represents the first full connection   |
|                             | layer.                                 |
+-----------------------------+----------------------------------------+
| chn                         | Channel numbers used to specify where  |
|                             | activation is extracted from           |
+-----------------------------+----------------------------------------+
| out                         | Output directory to save .stim.csv for |
|                             | top stimulus, and associated .act.hd5  |
|                             | file.                                  |
+-----------------------------+----------------------------------------+

Optional Arguments
------------------

+----------+----------------+
| Argument | Discription    |
+==========+================+
| cuda     | Use GPU or not |
+----------+----------------+

Outputs
=======

several stim.csv files that contain information of top stimuli of each
channel a .act.h5 file that contains raw activation of the top stimuli

Examples
========

Select top3 stimuli for 1st, 2nd, and 3rd channels of conv5 layer of
AlexNet respectively.

::

   dnn_topstim -net AlexNet -top 3 -stim examples.stim.csv -layer conv5 -chn 1 2 3 -out .
