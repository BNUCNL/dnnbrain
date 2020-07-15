Name
----

dnn_rsa - Characterise DNN activation's representation dissimilarity matrix (RDM).

Synopsis
--------

::

   dnn_rsa [-h] -act Activation [-layer Layer [Layer ...]]
           [-chn Channel [Channel ...]] [-dmask DnnMask] [-iteraxis Axis]
           [-cate Category] [-metric Metric] [-zscore] -out Output

Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| act                         | DNN activation file                    |
+-----------------------------+----------------------------------------+
| out                         | output filename with suffix as .rdm.h5 |
+-----------------------------+----------------------------------------+

Optional Arguments
~~~~~~~~~~~~~~~~~~

+---------------------+------------------------------------------------+
| Argument            | Discription                                    |
+=====================+================================================+
| layer               | layer names of interest |br|                   |
+---------------------+------------------------------------------------+
| chn                 | channel numbers of interest |br|               |
|                     | Default using all channels of each layer       |
|                     | specified by -layer.                           |
+---------------------+------------------------------------------------+
| dmask               | a .dmask.csv file in which layers of interest  |
|                     | are listed with their own channels, rows and   |
|                     | columns of interest.                           |
+---------------------+------------------------------------------------+
| iteraxis            | choices=(channel, row_col) |br|                |
|                     | Iterate along the specified axis. |br|         |
|                     | channel: Do rsa on each channel. |br|          |
|                     | row_col: Do rsa on each location (row_idx,     |
|                     | col_idx). |br|                                 |
|                     | default: Do rsa on the whole layer.            |
+---------------------+------------------------------------------------+
| cate                | a .stim.csv file which contains category       |
|                     | information (i.e. 'label' item) |br|           |
|                     | If used, do rsa category-wisely that average   |
|                     | activation pattern before calculating the      |
|                     | distance. And the row/column order of RDM is   |
|                     | organized from small to big according to the   |
|                     | 'label'. |br|                                  |
+---------------------+------------------------------------------------+
| metric              | Specify metric used to calculate distance. |br||
|                     | Default: euclidean                             |
+---------------------+------------------------------------------------+
| zscore              | Standardize feature values for each sample by  |
|                     | using zscore.                                  |
+---------------------+------------------------------------------------+


Outputs
-------

The output is a .rdm.h5 file, which contains each layer's RDM.  

Examples
--------

Calculate **euclidean** distance for each pair of stimuli using the activation pattern of each layer in **test.act.h5**. |br|
Save results to **out.rdm.h5**

::
 
   dnn_rsa -act test.act.h5 -out out.rdm.h5

.. |br| raw:: html

  <br/>
