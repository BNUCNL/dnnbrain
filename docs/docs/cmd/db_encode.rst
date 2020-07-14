Name
----

| db_encode - Use DNN activation to encode brain.
| The db_encode uses the representation from each layer in DNN
  to predict the response of a voxel in the human brain 
  by using voxel-wise encoding models (EM).

Synopsis
--------

db_encode -anal Analysis -act Activation [-layer Layer [Layer …]] [-chn Channel [Channel
…]] [-dmask DnnMask] [-iteraxis Axis] -resp Response -bmask BrainMask [-roi RoiName] 
-model Model [-scoring Scoring] [-cv FoldNumber] -out Output

Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+--------------------------------------------------+
| Argument                    | Discription                                      |
+=============================+==================================================+
| anal                        | Choices is ('uv', 'mv').                         |
|                             | 'uv' means doing univariate analysis;            |
|                             | 'mv' means doing multivariate analysis.          |
+-----------------------------+--------------------------------------------------+
| act                         | Path of a .act.h5 file which contains            |
|                             | activation information as the independent        |
|                             | variable.                                        |
+-----------------------------+--------------------------------------------------+
| resp                        | Path of a .roi.h5/.nii file which contains       |
|                             | brain response information as the dependent      |
|                             | variable. Note: if it is .nii file, -roi         |
|                             | will be ignored. "All voxels' activation         |
|                             | will be regarded as the "ground truth of         |
|                             | a regression task."                              |
+-----------------------------+--------------------------------------------------+
| bmask                       | Brain mask is used to extract activation locally.|
|                             | Only used when the response file is .nii file.   |
+-----------------------------+--------------------------------------------------+
| model                       | The model type to predict brain responses by dnn |
|                             | activation. Choices is ('glm', 'lasso'). You can |
|                             | use glm (general linear model) for regression or |
|                             | use lasso (lasso regression) for regression.     |
+-----------------------------+--------------------------------------------------+
| out                         | An output directory.                             |
+-----------------------------+--------------------------------------------------+

Optional Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+-----------------------------------------------------+
| Argument                    | Discription                                         |
+=============================+=====================================================+
| layer                       | Name of the target layer(s).Default is              |
|                             | all.E.g., ‘conv1’ represents the first              |
|                             | convolution layer, and ‘fc1’                        |
|                             | represents the first full connection layer.         |
+-----------------------------+-----------------------------------------------------+
| chn                         | Index of target channel(s).Default is               |
|                             | all.Channel index starts from 1.                    |
+-----------------------------+-----------------------------------------------------+
| dmask                       | Path of a .dmask.csv file in which                  |
|                             | detailed information of neuron(s) of                |
|                             | interest in DNN is specified. Argument              |
|                             | layer/chn and dmask are mutually                    |
|                             | exclusive. Provide only one of them if needed.      |
+-----------------------------+-----------------------------------------------------+
| iteraxis                    | Iterate along the specified axis.                   |
|                             | If -anal is uv: channel: Summarize the maximal      |
|                             | prediction score for each channel.                  |
|                             | row_col: Summarize the maximal prediction score     |
|                             | for each location (row_idx, col_idx).               |
|                             | default: Summarize the maximal prediction score     |
|                             | for the whole layer. If -anal is mva:               |
|                             | channel: Do mva using all units in each channel.    |
|                             | row_col: Do mva using all units in each location    |                                 
|                             | (row_idx, col_idx). default: Do mva using all units.|
+-----------------------------+-----------------------------------------------------+
| roi                         | Specify ROI names as the ground truth.              |
|                             | Default is using all ROIs in .roi.h5 file.          |
+-----------------------------+-----------------------------------------------------+
| Scoring                     | Model evaluation rules:                             |
|                             | correlation or sklearn scoring parameters.          |
|                             | Default is explained_variance.                      |
+-----------------------------+-----------------------------------------------------+
| cv                          | Cross validation fold number.                       |
|                             | Default is 3.                                       |
+-----------------------------+-----------------------------------------------------+


Outputs
-------

Arrays containing the prediction score and encoding accuracy maps of each layer.
Note：Different layers' output is stored in different folders.

Examples
--------

The example demonstrates the whole command of encoding.
DNN activation was used to predict brain response using GLM model 
in multivariate analysis.

::

   db_encode -anal mv -act AlexNet_relu_zscore_PCA-100.act.h5 -resp beta_rh_all_run.nii.gz -bmask VTC_mask_rh.nii.gz -model glm -scoring correlation -cv 10 -out AlexNet_relu_zscore_PCA-100_glm-corr_cv-10_VVA_rh

