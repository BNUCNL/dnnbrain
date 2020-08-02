Get Started
===========

Requirements
------------

-  python==3.6.x/3.7.x
-  1.16.5=<numpy<1.18
-  nibabel>=2.4.1
-  nipy>=0.4.2
-  h5py>=2.9.0
-  scikit-image>=0.15.0
-  scikit-learn>=0.19.1
-  scipy>=1.3.1
-  torch>=1.3.0
-  torchvision>=0.4.2
-  pillow>=6.0.0
-  opencv-python>=4.1.0.25
-  matplotlib>=2.2.2
-  cifti>=1.1

A easy way to get most of these requirements is to install a python
distribution (e.g.
`Anaconda <https://www.anaconda.com/products/individual>`__ is
recommended).

| After preparing these requirements, you can install DNNBrain by one of
  the following installation schemes. Finally, **do not forget to** read
  the `Configuration <#configuration>`__ and `Download DNN
  parameters <#download-dnn-parameters>`__.
| **Tip:** If the installation scheme you choose requires configuring
  environment variables, and you don’t know how to do it. Please skip to
  `Configuration <#configuration>`__

Install from source
-------------------

1. Get source code |br|
   Method1: Download source archive file (zip file) from
   `GitHub <https://github.com/BNUCNL/dnnbrain>`__ and unpack it. |br|
   Method2: Clone the DNNBrain repostitory by executing:
   ``git clone https://github.com/BNUCNL/dnnbrain.git``

2. Execute setup.py |br|
   Change to source directory (it should have the files README.md and setup.py). |br|
   Run ``python setup.py install``

| If you don’t have permission to install DNNBrain to the directory used
  by the above-mentioned command, you can install it to “the Python user
  install directory” by using ``–-user`` option:
| ``python setup.py install --user``
| Please ensure that “bin” or “Scripts” directory (named as
  *path_to_dnnbrain_bin*) in “the Python user install directory” is
  added to the environment variable PATH, otherwise, DNNBrain’s
  command-line interfaces can’t be searched by the shell.

| In addition, you can also install DNNBrain to the directory specified
  by yourself through the ``–-prefix`` or ``–-home`` options to setup.py.
| **Note**: If you want to install a package out of the standard python
  site-packages directory, you have to update two environment variables
  (PATH and PYTHONPATH) for your preferred directory. See `Installing
  Python Modules <https://docs.python.org/3/install/index.html>`__ for
  further details.

Install portable edition
------------------------

This installation scheme allows you put DNNBrain anywhere you like as
long as setting the two environment variables (PATH and PYTHONPATH)
appropriately. The scheme is described as below:

1. Get source code through methods mentioned at step 1 of `Install from
   source <#install-from-source>`__.
2. Add source directory (named as *path_to_dnnbrain*) to PYTHONPATH and
   “bin” directory under it (named as *path_to_dnnbrain_bin*) to PATH.

Configuration
-------------

No matter what installation scheme you choose, you have to set the
environment variable DNNBRAIN_DATA whose value is the absolute path of a
directory that used to store DNNBrain’s data, such as test data and
pretrained parameters of DNNs.

**Note:** If you work with DNNBrain on a public platform, such as a
Linux server, DNNBRAIN_DATA should be configured by the administrator of
the platform.

**Note:** The updates for PATH and PYTHONPATH are optional according to
the installation scheme’s requirement you choose.

| Assuming the value of DNNBRAIN_DATA is *path_to_dnnbrain_data*.
| Assuming the value which should be added to PATH is
  *path_to_dnnbrain_bin*.
| Assuming the value which should be added to PYTHONPATH is
  *path_to_dnnbrain*.

Linux \| MAC OS X
~~~~~~~~~~~~~~~~~

Open a terminal window and make sure that you know which “shell” you are
running (If not sure, try running the command ``printenv SHELL``).

For bash shell, run the following commands to set environment variables:

::

   echo 'export DNNBRAIN_DATA=path_to_dnnbrain_data' >> ~/.bashrc
   echo 'export PATH=path_to_dnnbrain_bin:$PATH' >> ~/.bashrc
   echo 'export PYTHONPATH=path_to_dnnbrain:$PYTHONPATH' >> ~/.bashrc

For tcsh/csh shell, run the following commands to set environment
variables:

::

   echo 'set DNNBRAIN_DATA=path_to_dnnbrain_data' >> ~/.cshrc
   echo 'set PATH=(path_to_dnnbrain_bin $PATH)' >> ~/.cshrc
   echo 'set PYTHONPATH=(path_to_dnnbrain $PYTHONPATH)' >> ~/.cshrc

Alternatively, you can also set environment variables by editing .bashrc
or .cshrc file in a text editor.

You have to make these changes be effective by openning a new terminal
window or “source” the .bashrc/.cshrc file in the current terminal
window.

Windows
~~~~~~~

**Note**: DNNBrain scripts in *path_to_dnnbrain_bin* are only executable in UNIX-like shell, such as `git bash <https://gitforwindows.org/>`__.

Open the environment variable editor which can be found under “Windows
Menu —> Control Panel —> Advanced System Settings —> Environment
Variables”. Then:

-  Create a new environment variable named DNNBRAIN_DATA with value as
   *path_to_dnnbrain_data*.
-  Add *path_to_dnnbrain_bin* to PATH.
-  Add *path_to_dnnbrain* to PYTHONPATH (If PYTHONPATH is not existed,
   create a new one).

Separate multiple paths with semicolons (;).

Alternatively, you can use the following commands in PowerShell to complete the above 3 steps (Please replace the path between asterisks with the real path, for example, the \*path_to_dnnbrain\* should be replaced with the \'F:\\Python3.6.5\\Lib\\site-packages\\dnnbrain\')

::

    $new_path = *path_to_dnnbrain_data*
    [environment]::SetEnvironmentvariable('DNNBRAIN_DATA', $new_path, "User")

    $old_path = [environment]::GetEnvironmentvariable("PATH", "User")
    $path_to_dnnbrain_data_bin = *path_to_dnnbrain_data_bin*
    $new_path=$old_path,$path_to_dnnbrain_data_bin -Join ";"
    [environment]::SetEnvironmentvariable("PATH", $new_path, "User")

    $old_path = [environment]::GetEnvironmentvariable("PYTHONPATH", "User")
    $path_to_dnnbrain = *path_to_dnnbrain*
    $new_path=$old_path,$path_to_dnnbrain -Join ";"
    [environment]::SetEnvironmentvariable("PYTHONPATH", $new_path, "User")

Download DNN parameters
-----------------------

The pretrained parameters are always necessary and they should be placed
in the “**models**” directory under the DNNBRAIN_DATA.

| The pretrained parameters of DNNs supported by DNNBrain are shown as
  below. You can download preferred DNN parameters by clicking
  corresponding filenames **(Make sure to rename the downloaded file as
  its filename used here)**.
| `alexnet.pth <https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth>`__
  \|
  `vgg11.pth <https://download.pytorch.org/models/vgg11-bbd30ac9.pth>`__
  \|
  `vgg_face_dag.pth <http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth>`__
  \|
  `vgg19_bn.pth <https://download.pytorch.org/models/vgg19_bn-c79401a0.pth>`__
  \|
  `googlenet.pth <https://download.pytorch.org/models/googlenet-1378be20.pth>`__
  \|
  `resnet152.pth <https://download.pytorch.org/models/resnet152-b121ed2d.pth>`__


.. |br| raw:: html

   <br />

