Name
----

dnn_mask - Extract DNN activation for layer, channel, row and column of
interest.

Synopsis
--------

dnn_mask -act Activation [-layer Layer [Layer …]] [-chn Channel [Channel
…]] [-dmask DnnMask] -out Output

Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| act                         | Path of a .act.h5 file which contains  |
|                             | activation information to extract      |
|                             | interested activation.                 |
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

Outputs
-------

An .act.h5 file containing the interested activation that can be read
and saved with the module dnnbrain.io.fileio.ActivationFile.

Examples
--------

These examples demonstrate the interested activation extraction
function. Activation provided by test.act.h5 was extracted and finally
saved in the dnn_mask.act.h5 file.

::

   # Asserting target layers using the -layer argument
   dnn_mask -act ./test.act.h5 -layer conv5 -chn 1 2 3 -stim ./test.stim.csv -out ./dnn_mask1.act.h5

::

   # Asserting target layers using the -dmask argument
   dnn_mask -act ./test.act.h5 -dmask ./test.dmask.csv -stim ./test.stim.csv -out ./dnn_mask2.act.h5
