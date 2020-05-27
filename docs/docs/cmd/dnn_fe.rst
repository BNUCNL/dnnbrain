Name
----

| dnn_fe - Extract features of DNN activation.
| The dnn_fe using methods to reduct feature dimension, clustering
  several main components of the DNN high dimensional activation as
  required

Synopsis
--------

dnn_fe -act Activation [-layer Layer [Layer …]] [-chn Channel [Channel
…]] [-dmask DnnMask] -meth Method [-axis Axis] -out Output

Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| act                         | Path of a .act.h5 file which contains  |
|                             | activation information to extract      |
|                             | features.                              |
+-----------------------------+----------------------------------------+
| meth                        | Method of feature extraction.Enter two |
|                             | parameters in order (‘Method’,         |
|                             | ‘N_feature’).The first means the       |
|                             | method choosing from (‘hist’, ‘psd’,   |
|                             | ‘pca’).The second is used to specify   |
|                             | the number of features we will use.    |
+-----------------------------+----------------------------------------+
| out                         | Output path with a suffix as .act.h5.  |
+-----------------------------+----------------------------------------+

Optional Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| layer                       | Name of the target layer(s).Default is |
|                             | all.E.g., ‘conv1’ represents the first |
|                             | convolution layer, and ‘fc1’           |
|                             | represents the first full connection   |
|                             | layer.                                 |
+-----------------------------+----------------------------------------+
| chn                         | Index of target channel(s).Default is  |
|                             | all.Channel index starts from 1.       |
+-----------------------------+----------------------------------------+
| dmask                       | Path of a .dmask.csv file in which     |
|                             | detailed information of neuron(s) of   |
|                             | interest in DNN is specified.Argument  |
|                             | layer/chn and dmask are mutually       |
|                             | exclusive. Provide only one of them if |
|                             | needed.                                |
+-----------------------------+----------------------------------------+
| axis                        | Axis for feature extraction.Default is |
|                             | extracting features from the whole     |
|                             | layer.chn: axis in chn; row_col: axis  |
|                             | in specific row_col unit in a feature  |
|                             | map;                                   |
+-----------------------------+----------------------------------------+

Outputs
-------

An .act.h5 file containing the feature extracted activation that can be
read and saved with the module dnnbrain.io.fileio.ActivationFile.

Examples
--------

These examples demonstrate the activation feature extraction function.
Activation provided by test.act.h5 was extracted feature and finally
saved in the dnn_fe.act.h5 file.

::

   # Asserting target layers using the -layer argument
   dnn_fe -act ./test.act.h5 -layer conv1 conv5_relu fc2_relu -stim ./test.stim.csv -meth pca 3 -axis chn -out ./dnn_fe1.act.h5

::

   # Asserting target layers using the -dmask argument
   dnn_fe -act ./test.act.h5 -dmask ./test.dmask.csv -stim ./test.stim.csv -meth pca 3 -axis chn -out ./dnn_fe2.act.h5

::

   # Not asserting layers which means target layers and chns are all
   dnn_fe -act ./test.act.h5 -stim ./test.stim.csv -meth pca 3 -axis chn -out ./dnn_fe3.act.h5
