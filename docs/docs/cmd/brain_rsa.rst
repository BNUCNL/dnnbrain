Name
----

brain_rsa - Characterise brain activation's representation dissimilarity matrix (RDM).

Synopsis
--------

::

   brain_rsa [-h] -nif NeuroImageFile -bmask BrainMask
             [-roi ROI [ROI ...]] [-cate Category] [-metric Metric]
             [-zscore] -out Output

Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| nif                         | brain activation image                 |
+-----------------------------+----------------------------------------+
| bmask                       | brain mask image |br|                  |
|                             | Each non-zero value will be regarded   |
|                             | as a ROI label. |br|                   |
|                             | The activation pattern in each ROI     |
|                             | will be used to calculate RDM.         |
+-----------------------------+----------------------------------------+
| out                         | output filename with suffix as .rdm.h5 |
+-----------------------------+----------------------------------------+

Optional Arguments
~~~~~~~~~~~~~~~~~~

+---------------------+------------------------------------------------+
| Argument            | Discription                                    |
+=====================+================================================+
| roi                 | Specify which ROI labels in bmask              |
|                     | will be used; |br|                             |
|                     | Default using all labels.                      |
+---------------------+------------------------------------------------+
| cate                | a .stim.csv file which contains category       |
|                     | information (i.e. 'label' item) |br|           |
|                     | If used, do rsa category-wisely that average   |
|                     | activation pattern before calculating the      |
|                     | distance. And the row/column order of RDM is   |
|                     | organized from small to big according to the   |
|                     | 'label'. |br|                                  |
|                     | Note: the 'label' here is an item in the       |
|                     | .stim.csv file rather than the label in '-roi' |
|                     | option!                                        |
+---------------------+------------------------------------------------+
| metric              | Specify metric used to calculate distance. |br||
|                     | Default: euclidean                             |
+---------------------+------------------------------------------------+
| zscore              | Standardize feature values for each sample by  |
|                     | using zscore.                                  |
+---------------------+------------------------------------------------+


Outputs
-------

The output is a .rdm.h5 file, which contains each ROI's RDM.  

Examples
--------

Each volume in **nif.nii.gz** is an activation map of each stimulus. |br|
Calculate **correlation** distance for each pair of stimuli using the activation pattern of **ROI1** (voxels with label 1 in **bmask.nii.gz**) and **ROI3** (voxels with label 3 in **bmask.nii.gz**) successively. |br|
Save results to **out.rdm.h5**

::
 
   brain_rsa -nif nif.nii.gz -bmask bmask.nii.gz -roi 1 3 -metric correlation -out out.rdm.h5

.. |br| raw:: html

  <br/>
