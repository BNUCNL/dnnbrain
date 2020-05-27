Name
----

db_info - Provide important information of specific dnn or file

Synopsis
--------

db_info filename/netname

Arguments
---------

Required Position Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| name                        | Filename or Netname.Filename only      |
|                             | support suffix with                    |
|                             | ‘.stim.csv’,‘.dmask.csv’,‘.act.h5’,‘.r |
|                             | oi.h5’,‘.rdm.h5’.Remember              |
|                             | to run this command in the right path  |
|                             | or it can’t find your file.Netname     |
|                             | only support ‘AlexNet’, ‘VggFace’,     |
|                             | ‘Vgg11’                                |
+-----------------------------+----------------------------------------+

Examples
--------

Check for dnn information
~~~~~~~~~~~~~~~~~~~~~~~~~

Provide dnn’s structure parameters and layer to location information

::

   db_info AlexNet

The output information is displayed as below:

.. raw:: html

   <center>

|AlexNet|

.. raw:: html

   </center>

Check for file information
~~~~~~~~~~~~~~~~~~~~~~~~~~

.stim.csv
^^^^^^^^^

Provide information of stimuli type, path, data types and number of
stimuli.

::

   db_info test.stim.csv

The output information is displayed as below:

.. raw:: html

   <center>

|stim|

.. raw:: html

   </center>

.dmask.csv
^^^^^^^^^^

Provide information of mask layer, chn, row and column.

::

   db_info test.dmask.csv

The output information is displayed as below:

.. raw:: html

   <center>

|dmask|

.. raw:: html

   </center>

.act.h5
^^^^^^^

Provide activation shape in different layers and its statistical
information.

::

   db_info test.act.h5

The output information is displayed as below:

.. raw:: html

   <center>

|act|

.. raw:: html

   </center>

.roi.h5
^^^^^^^

Provide ROI names, data shape of brain response and its statistical
information.

::

   db_info test.roi.h5

The output information is displayed as below:

.. raw:: html

   <center>

|roi|

.. raw:: html

   </center>

.rdm.h5
^^^^^^^

Provide representation distance matrices (RDMs) shape for DNN activation
and brain activation and their statistical information.

::

   # Checking information for RDM type of brain
   db_info brain.rdm.h5

The output information is displayed as below:

.. raw:: html

   <center>

|brdm|

.. raw:: html

   </center>

::

   # Checking information for RDM type of dnn
   db_info dnn.rdm.h5

The output information is displayed as below:

.. raw:: html

   <center>

|drdm|

.. raw:: html

   </center>

.. |AlexNet| image:: ../../img/AlexNet_info.png
.. |stim| image:: ../../img/stim_info.png
.. |dmask| image:: ../../img/dmask_info.png
.. |act| image:: ../../img/act_info.png
.. |roi| image:: ../../img/roi_info.png
.. |brdm| image:: ../../img/brain_rdm_info.png
.. |drdm| image:: ../../img/dnn_rdm_info.png

