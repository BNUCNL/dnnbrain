Name
----

brain_roi - Extract brain ROI signals

Synopsis
--------

::

   brain_roi [-h] -nif BrainAct -mask Mask [-method Method] -out Output

Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| nif                         | brain activation image                 |
+-----------------------------+----------------------------------------+
| mask                        | brain mask image |br|                  |
|                             | Each non-zero value will be regarded   |
|                             | as a ROI label.                        |
+-----------------------------+----------------------------------------+
| out                         | Output file with suffix as .roi.h5     |
+-----------------------------+----------------------------------------+

Optional Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| method                      | choices=(mean, max, std) |br|          |
|                             | Method to summary signals in ROI.      |
|                             | (default: mean)                        |
+-----------------------------+----------------------------------------+


Outputs
-------

The output is a .roi.h5 file, which contains each ROI's signal.

Examples
--------

Calculate each ROI's *mean* signal in *nif.nii.gz*. The ROIs are specified by ROI labels in *mask.nii.gz*. Save results to *out.roi.h5*.

::
 
   brain_roi -nif nif.nii.gz -mask mask.nii.gz -method mean -out out.roi.h5

.. |br| raw:: html

  <br/>
