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

## Install from source
1. Get source code  
Method1: Download source archive file (zip file) from [GitHub](https://github.com/BNUCNL/dnnbrain) and unpack it.  
Method2: Clone the DNNBrain repostitory by executing:  
```$ git clone https://github.com/BNUCNL/dnnbrain.git```
2. Execute setup.py  
Change to source directory (it should have the files README.md and setup.py).  
Run ```python setup.py install```  

If you don’t have permission to install DNNBrain on the user-shared space, you can install it to your own user-specific space by using --user option:  
```python setup.py install --user```  

Or another directory you preferred by using the --prefix or --home options to setup.py.  

**Note**: If you want to install a package out of the standard python site-packages directory, you have to set PYTHONPATH variable for your preferred directory. See [Installing Python Modules](https://docs.python.org/3/install/index.html) for further details.
## Configuration
**Note:** If you work with DNNBrain on a public platform, such as a Linux server, this step should be configured by the administrator of the platform.  

Before using DNNBrain, you need to define an environment variable called DNNBRAIN_DATA which is set to the location that used to store DNNBrain's data, such as test data and offline DNN model files. The model files are necessary and they should be placed in the 'models' directory under the DNNBRAIN_DATA. (You can get the model files from [ToBeConfirmed])