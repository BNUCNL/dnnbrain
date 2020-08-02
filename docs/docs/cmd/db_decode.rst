Name
----

| db_decode - Decode DNN activation from brain response.
| The db_decode decodes the response of a voxel in the human brain 
  to the representation from each layer in DNN.  

Synopsis
--------

::

   db_decode  [-h] -anal Analysis -resp Response [-bmask BrainMask]
              [-roi RoiName [RoiName ...]] -act Activation
              [-layer Layer [Layer ...]] [-chn Channel [Channel ...]]
              [-dmask DnnMask] -model Model [-cv CrossValidationFoldNumber]
              -out Output
Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+--------------------------------------------------+
| Argument                    | Discription                                      |
+=============================+==================================================+
| anal                        | choices=(uv, mv) |br|                            |
|                             | uv: Do univariate mapping |br|                   |
|                             | mv: Do multivariate mapping                      |
+-----------------------------+--------------------------------------------------+
| act                         | Path of a .act.h5 file which contains            |
|                             | activation information.                          |
+-----------------------------+--------------------------------------------------+
| resp                        | Path of a .roi.h5/.nii file which contains       |
|                             | brain response information. |br|                 |
|                             | If it is .nii file, -roi will be ignored. |br|   |
|                             | If it is .roi.h5 file, -bmask will be ignored.   |
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

+------------+-------------------------------------------------------------------+
| Argument   | Discription                                                       |
+============+===================================================================+
| layer      | Name of the target layer(s). Default is all.                      |
+------------+-------------------------------------------------------------------+
| chn        | Index of target channel(s). Default is                            |
|            | all.Channel index starts from 1.                                  |
+------------+-------------------------------------------------------------------+
| dmask      | Path of a .dmask.csv file in which                                |
|            | detailed information of neuron(s) of                              |
|            | interest in DNN is specified. Argument                            |
|            | layer/chn and dmask are mutually                                  |
|            | exclusive. Provide only one of them if needed.                    |
+------------+-------------------------------------------------------------------+
| bmask      | Brain mask is used to extract activation locally. |br|            |
|            | Only used when the response file is .nii file. |br|               |
|            | If not given, the whole brain response will be used.              |
+------------+-------------------------------------------------------------------+
| roi        | Specify ROI names for decoding. |br|                              |
|            | Default is using all ROIs in .roi.h5 file.                        |
+------------+-------------------------------------------------------------------+
| cv         | Cross validation fold number.                                     |
|            | Default is 3.                                                     |
+------------+-------------------------------------------------------------------+


Outputs
-------

Arrays containing the prediction score of each layer.
Note：Different layers' output is stored in different folders.

Examples
--------

Decode brain response(test.nii.gz) to DNN activation(test.act.h5) using GLM model 
with 10 cross validation fold numbers in multivariate analysis.

::

   db_decode -anal mv -act test.act.h5 -resp test.nii.gz -model glm -cv 10 -out test_glm_cv-10 


.. |br| raw:: html

    <br/>