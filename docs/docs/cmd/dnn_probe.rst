Name
----

dnn_probe - Probe the ability of DNN activation to predict behavior.

Synopsis
--------

::

   dnn_probe [-h] -anal Analysis -act Activation [-layer Layer [Layer …]]
             [-chn Channel [Channel…]] [-dmask DnnMask] [-iteraxis Axis]
             -stim Stimulus -beh Behavior -model Model [-cv FolderNumber] -out Output

Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| anal                        |choices=('uv', 'mv') |br|               |
|                             |'uv': univariate analysis |br|          |
|                             |'mv': multivariate analysis             |
+-----------------------------+----------------------------------------+
| act                         | Path of a .act.h5 file which contains  |
|                             | activation information to extract      |
|                             | features.                              |
+-----------------------------+----------------------------------------+
| stim                        | a .stim.csv file which contains        |
|                             | stimulus information.                  |
+-----------------------------+----------------------------------------+
| beh                         | Specify behaviors as the groud truth   |
+-----------------------------+----------------------------------------+
| model                       | Method of analysis model. Enter one    |
|                             | parameter 'Model', choose from: |br|   |
|                             | 'glm': general linear model  |br|      |
|                             | 'lasso': lasso regression    |br|      |
|                             | 'svc': support vector machine |br|     |
|                             | 'lrc': logistic regression |br|        |
|                             | 'corr': pearson correlation |br|       |
+-----------------------------+----------------------------------------+
| out                         | An output directory                    |
+-----------------------------+----------------------------------------+

Optional Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| layer                       | Name of the target layer(s). Default is|
|                             | all.                                   |
+-----------------------------+----------------------------------------+
| chn                         | Index of target channel(s). Default is |
|                             | all. Channel index starts from 1.      |
+-----------------------------+----------------------------------------+
| dmask                       | Path of a .dmask.csv file in which     |
|                             | detailed information of neuron(s) of   |
|                             | interest in DNN is specified. |br|     |
|                             | Argument layer/chn and dmask are       |
|                             | mutually exclusive. |br|               | 
|                             | Provide only one of them if needed.    |
+-----------------------------+----------------------------------------+
| iteraxis                    | Axis for model iteration.  |br|        |
|                             | Default for                            |
|                             | 'uv' analysis is to summarize the      |
|                             | maximal prediction score for the whole |
|                             | layer. |br|                            |
|                             | Default for 'mv' analysis is to do     |
|                             | analysis using all units in layer. |br||
|                             | If -anal is 'uv':  |br|                |
|                             | 'channel': Summarize the maximal       |
|                             | prediction score for each channel. |br||
|                             | 'row_col': Summarize the maximal       |
|                             | prediction score for each location     |
|                             | (row_idx, col_idx). |br|               |
|                             | If -anal is 'mv': |br|                 |
|                             | 'channel': Do mva using all units in   |
|                             | each channel. |br|                     | 
|                             | 'row_col': Do mva using all units in   | 
|                             | each location (row_idx, col_idx). |br| |
|                             |                                        |
+-----------------------------+----------------------------------------+
| cv                          | Cross validation fold number.          |
+-----------------------------+----------------------------------------+


Outputs
-------

An output directory containing the analysis result files. For each layer,
analysis result is stored as a .csv file in a subfold named after the layer.  

Examples
--------

Train a logistic regression model for each layer in Test.act.h5 to decode its representation to category labels in Test.stim.csv. The accuracy of the model is evaluated with a 10-fold cross validation, and the outputs are saved in Test_lrc_label_cv10.

::
 
    dnn_probe -anal mv -act Test.act.h5 -stim Test.stim.csv -beh label -model lrc -cv 10 -out Test_lrc_label_cv10

.. |br| raw:: html

  <br/>
