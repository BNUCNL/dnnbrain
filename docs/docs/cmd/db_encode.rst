Name
----

| db_encode - Use DNN activation to encode brain.
| The db_encode uses the representation from each layer in DNN
  to predict the response of a voxel in the human brain 
  by using voxel-wise encoding models (EM).

Synopsis
--------

::

   db_encode  [-h] -anal Analysis -act Activation
              [-layer Layer [Layer ...]] [-chn Channel [Channel ...]]
              [-dmask DnnMask] [-iteraxis Axis] -resp Response
              [-bmask BrainMask] [-roi RoiName [RoiName ...]] -model Model
              [-scoring Scoring] [-cv CrossValidationFoldNumber] -out Output

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
| layer                       | Name of the target layer(s).Default is all.         |
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
| bmask                       | Brain mask is used to extract activation locally.   |
|                             | Only used when the response file is .nii file.      |
|                             | If not given, the whole brain response will be used.|
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

Arrays containing the prediction score of each layer.
Note：Different layers' output is stored in different folders.

Examples
--------

DNN activation(test.act.h5) was used to predict brain response(test.nii.gz) using GLM model 
in multivariate analysis. The example uses the scoring of correlation with 10 cross validation fold numbers.


::

   db_encode -anal mv -act test.act.h5 -resp test.nii.gz -model glm -scoring correlation -cv 10 
   -out test_glm-corr_cv-10
