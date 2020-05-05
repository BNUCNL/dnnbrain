# Installation
## Requirements
- python>=3.6.7
- numpy>=1.16.5
- nibabel>=2.4.1
- h5py>=2.9.0
- pandas>=0.24.2
- scikit-image>=0.15.0
- scikit-learn>=0.19.1
- scipy>=1.3.1
- pytorch>=1.3.0
- pillow>=6.0.0
- opencv-python>=4.1.0.25
- matplotlib>=2.2.2
- cifti>=1.1

A easy way to get most of these requirements is to install a python distribution (e.g. [Anaconda](https://www.anaconda.com/products/individual) is recommended).

## Install from source
1. Get source code  
Method1: Download source archive file (zip file) from [GitHub](https://github.com/BNUCNL/dnnbrain) and unpack it.  
Method2: Clone the DNNBrain repostitory by executing:  
```$ git clone https://github.com/BNUCNL/dnnbrain.git```
2. Execute setup.py  
Change to source directory (it should have the files README.md and setup.py).  
Run ```python setup.py install```  

If you don’t have permission to install DNNBrain to the directory used by the above-mentioned command, you can install it to "the Python user install directory" by using --user option:  
```python setup.py install --user```  
Please ensure that "bin" directory in "the Python user install directory" is updated to the PATH environment, otherwise, DNNBrain's command-line interfaces can't be searched by the shell.

In addition, you can also install DNNBrain to the directory specified by yourself through the --prefix or --home options to setup.py.  
**Note**: If you want to install a package out of the standard python site-packages directory, you have to update two environment variables (PATH and PYTHONPATH) for your preferred directory. See [Installing Python Modules](https://docs.python.org/3/install/index.html) for further details.

## Configuration
**Note:** If you work with DNNBrain on a public platform, such as a Linux server, this step should be configured by the administrator of the platform.  

Before using DNNBrain, you need to set an environment variable named DNNBRAIN_DATA whose value is the absolute path of a directory that used to store DNNBrain's data, such as test data and offline DNN model files. The model files are necessary and they should be placed in the 'models' directory under the DNNBRAIN_DATA.

### Set environment variable
Assuming the value of DNNBRAIN_DATA is path_to_dnnbrain_data.

#### Linux | MAC OS X
Open a terminal window and make sure that you know which "shell" you are running (If not sure, try running the command ```printenv SHELL```).

For bash shell, run the following command to set the DNNBRAIN_DATA:
```
echo 'export DNNBRAIN_DATA=path_to_dnnbrain_data' >> ~/.bashrc
```

For tcsh/csh shell, run the following command to set the DNNBRAIN_DATA:
```
echo 'set DNNBRAIN_DATA=path_to_dnnbrain_data' >> ~/.cshrc
```

Alternatively, you can also set the DNNBRAIN_DATA by editing .bashrc or .cshrc file in a text editor.

You have to make these changes be effective by openning a new terminal window or "source" the .bashrc/.cshrc file in the current terminal window.

#### Windows
Open the environment variable editor which can be found under "Windows Menu —> Control Panel —> Advanced System Settings —> Environment Variables". Then, create a new environment variable named DNNBRAIN_DATA with value as path_to_dnnbrain_data.

### Get DNN model file
You can get model files by using dnn_download which is one of the command-line interfaces of DNNBrain. Such as:
```
dnn_download AlexNet
```
