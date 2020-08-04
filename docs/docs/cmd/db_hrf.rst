Name
----

db_hrf - Convolve DNN activation with SPM canonical hemodynamic response function. And match it with the time points of Brain activation.

Synopsis
--------

::

   db_hrf [-h] -act Activation [-layer Layer [Layer ...]]
          [-chn Channel [Channel ...]] [-dmask DnnMask] -stim Stimulus
          -tr TR -n_vol N_volume [-ops Ops] -out Output

Arguments
---------

Required Arguments
~~~~~~~~~~~~~~~~~~

+-----------------------------+----------------------------------------+
| Argument                    | Discription                            |
+=============================+========================================+
| act                         | DNN activation file                    |
+-----------------------------+----------------------------------------+
| stim                        | a .stim.csv file which contains onsets |
|                             | and durations                          |
+-----------------------------+----------------------------------------+
| tr                          | repetition time of BOLD signal         |
|                             | acquisition                            |
+-----------------------------+----------------------------------------+
| n_vol                       | the number of volumes of BOLD signal   |
+-----------------------------+----------------------------------------+
| out                         | output filename with suffix as .act.h5 |
+-----------------------------+----------------------------------------+

Optional Arguments
~~~~~~~~~~~~~~~~~~

+---------------------+------------------------------------------------+
| Argument            | Discription                                    |
+=====================+================================================+
| layer               | layer names of interest                        |
+---------------------+------------------------------------------------+
| chn                 | channel numbers of interest |br|               |
|                     | Default using all channels of each layer       |
|                     | specified by -layer.                           |
+---------------------+------------------------------------------------+
| dmask               | a .dmask.csv file in which layers of interest  |
|                     | are listed with their own channels, rows and   |
|                     | columns of interest.                           |
+---------------------+------------------------------------------------+
| ops                 | choices=(10, 100, 1000) |br|                   |
|                     | oversampling number per second (default: 100)  |
+---------------------+------------------------------------------------+


Outputs
-------

The output is a .act.h5 file.

Examples
--------

Convolve activation of layer **fc3** in **test.act.h5** with SPM canonical hemodynamic response function according to onsets and durations in **test.stim.csv**. And match it with the **152** time points of Brain activation with repetition time as **2 seconds**. Save results to **out.act.h5**.

::
 
   db_hrf -act test.act.h5 -layer fc3 -stim test.stim.csv -tr 2 -n_vol 152 -out out.act.h5

.. |br| raw:: html

  <br/>
