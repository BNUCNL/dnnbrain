Name
----

db_corr - Correlate DNN activation with brain response

Synopsis
--------

::

   db_corr [-h] -act Activation [-layer Layer [Layer ...]]
           [-chn Channel [Channel ...]] [-dmask DnnMask] [-iteraxis Axis]
           -resp Response [-bmask BrainMask] [-roi RoiName [RoiName ...]]
           -out Output

Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~~

+-------------------+--------------------------------------------------+
| Argument          | Discription                                      |
+===================+==================================================+
| act               | DNN activation file                              |
+-------------------+--------------------------------------------------+
| resp              | a .roi.h5/.nii file |br|                         |
|                   | If it is .nii file, -roi will be ignored. |br|   |
|                   | All voxels' activation will be a correlate.      |
+-------------------+--------------------------------------------------+
| out               | output directory.                                |
+-------------------+--------------------------------------------------+

Optional Arguments
~~~~~~~~~~~~~~~~~~

+-------------------+-----------------------------------------------------+
| Argument          | Discription                                         |
+===================+=====================================================+
| layer             | layer names of interest                             |
+-------------------+-----------------------------------------------------+
| chn               | channel numbers of interest |br|                    |
|                   | Default is using all channels of each layer         |
|                   | specified by -layer.                                |
+-------------------+-----------------------------------------------------+
| dmask             | a .dmask.csv file in which layers of interest are   |
|                   | listed with their own channels, rows and columns of |
|                   | interest.                                           |
+-------------------+-----------------------------------------------------+
| iteraxis          | choices=(channel, row_col) |br|                     |
|                   | Iterate along the specified axis. |br|              |
|                   | channel: Summarize the maximal pearson r for each   |
|                   | channel. |br|                                       |
|                   | row_col: Summarize the maximal pearson r for each   |
|                   | position (row_idx, col_idx). |br|                   |
|                   | default: Summarize the maximal pearson r for the    |
|                   | whole layer.                                        |
+-------------------+-----------------------------------------------------+
| bmask             | Brain mask is used to extract activation            |
|                   | locally. |br|                                       |
|                   | Voxels with non-zero value will be regarded as      |
|                   | correlates |br|                                     |
|                   | Only used when the response file is .nii file.      |
+-------------------+-----------------------------------------------------+
| roi               | Specify ROI names as the correlates. |br|           |
|                   | Default is using all ROIs in .roi.h5 file.          |
+-------------------+-----------------------------------------------------+

Outputs
-------

| Maximal pearson r and its location will be saved.
| Different layers' output is stored in different folders.

Examples
--------

| Calculate pearson r correlation between DNN unit and brain voxel.
| For each voxel, find the unit with the maximal correlation within each layer.
| The DNN units are from layer **conv5** and **fc1**.
| The brain voxels are those with non-zero value in **bmask.nii.gz**.
| Maximal pearson r and its location will be saved at **out_dir**


::

   db_corr -act test.act.h5 -layer conv5 fc1 -resp resp.nii.gz -bmask bmask.nii.gz -out out_dir

.. |br| raw:: html

   <br/>