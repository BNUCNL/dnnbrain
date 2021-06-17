Tutorial
========

We use the AlexNet model and BOLD5000 dataset [#0]_ to demonstrate the versatility and usability of DNNBrain in characterizing the DNN and in
examining the correspondences between the DNN and the brain. The results are also used in our `paper <https://doi.org/10.3389/fncom.2020.580632>`__, and more description about the model and dataset can be found at the 'Methods' part of this paper.

.. topic:: Data:

   | All the data you'll need for this tutorial is uploaded to `OSF <https://osf.io/gzwav/>`__. Here are some descriptions and specific download links.
   | The stimulus images are available by clicking `here <https://osf.io/hy5m7/download>`__, and a :doc:`.stim.csv <../user_guide/file_format>` file named as `all_5000scenes.stim.csv <https://osf.io/7c8th/download>`__ is used to tell DNNBrain where and what the inputs are.
   | The BOLD response maps for each image are available at `here <https://osf.io/ube86/download>`__.
   | The VTC mask is available by clicking `here <https://osf.io/w7ved/download>`__.
   | More details about the data can be found in `readme.txt <https://osf.io/3ng7k/download>`__.

.. topic:: Tutorials:

   - In :doc:`Scan DNN <scan_DNN>` tutorial, we extract and display feature maps of three images for each convolutional layer after ReLU.
   
   - In :doc:`Probe DNN <probe_DNN>` tutorial, we reveal animate information presented in DNN layers.
   
   - In :doc:`Map between DNN and brain <map_between_DNN_and_brain>` tutorial, we examine how well the representation from each layer predict the response of a voxel in the brain by using voxel-wise encoding models. In addition, we also use representational similarity analysis to characterize the link between the representations of DNN and brain.
   
   - In :doc:`Visualize DNN <visualize_DNN>` tutorial, We use three visualization ways to examine the stimulus features that an artificial neuron prefers.

.. topic:: References:

    .. [#0] Chang, N., Pyles, J. A., Marcus, A., Gupta, A., Tarr, M. J., and Aminoff, E. M. (2019). BOLD5000, a public fMRI dataset while viewing 5000 visual images. Sci. data 6, 49. doi:10.1038/s41597-019-0052-3.