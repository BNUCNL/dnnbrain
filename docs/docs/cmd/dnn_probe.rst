Name
----

dnn_probe - Probe the ability of DNN activation to predict behavior.

Synopsis
--------

dnn_probe -anal Analysis -act Activation [-layer Layer [Layer …]] [-chn Channel [Channel
…]] [-dmask DnnMask] [-iteraxis Axis] -stim Stimulus -beh Behavior -model Model [-cv FolderNumber] -out Output

Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| anal                        | Method of analysis. Enter one parameter|
|                             | 'Analysis', choose from ('uv','mv').   |
|                             | 'uv' means univariate analysis, and    |
|                             | 'mv' means multivariate analysis       |
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
|                             | parameter 'Model', choose from ('glm', |
|                             |'lasso','svc','lrc','corr').            |
+-----------------------------+----------------------------------------+
| out                         | An output directory                    |
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
| iteraxis                    | Axis for model iteration. Default for  |
|                             | 'uv' analysis is to summarize the      |
|                             | maximal prediction score for the whole |
|                             | layer. Default for 'mv' analysis is to |
|                             | do analysis using all units in layer.  |
+-----------------------------+----------------------------------------+
| cv                          | Cross validation fold number.          |
+-----------------------------+----------------------------------------+


Outputs
-------

An output directory containing the analysis result files. For each layer,
analysis result is stored as a .csv file in a subfold named after the layer.  

Examples
--------

Train a logistic regression model on the artificial representation from each layer to decode the stimulus category. The accuracy of the model is evaluated with a 10-fold cross validation.

::
 
    dnn_probe -anal mv -act AlexNet_PCA-100.act.h5 -stim all_5000scenes.stim.csv -beh label -model lrc -cv 10 -out AlexNet_lrc_label_cv10
    dnn_probe -anal mv -act AlexNet_relu_PCA-100.act.h5 -stim all_5000scenes.stim.csv -beh label -model lrc -cv 10 -out AlexNet_lrc_label_cv10
