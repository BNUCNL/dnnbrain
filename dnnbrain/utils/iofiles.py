import os
import scipy.io
import numpy as np
import pandas as pd

try:
    from PIL import Image
except ModuleNotFoundError:
    raise Exception('Please install pillow in your work station')

try:
    import torch
    import torchvision
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    raise Exception('Please install pytorch and torchvision in your work station')
    
try:
    import nibabel as nib
    import cifti

except ModuleNotFoundError:
    raise Exception('Please install nibabel and cifti in your work station')

DNNBRAIN_MODEL_DIR = os.environ['DNNBRAIN_MODEL_DIR']


class PicDataset(Dataset):
    """
    Build a dataset to load pictures
    """
    def __init__(self, csv_file, transform=None):
        """
        Initialize PicDataset
        
        Parameters:
        ------------
        csv_file[str]:  table contains picture names, conditions and picture onset time.
                        This csv_file helps us connect cnn activation to brain images.
                        Please organize your information as:
                                     
                        [PICDIR]
                        stimID     condition   onset(optional) measurement(optional)
                        face1.png  face        1.1             3
                        face2.png  face        3.1             5
                        scene1.png scene       5.1             4
        
        transform[callable function]: optional transform to be applied on a sample.
        """
        self.csv_file = pd.read_csv(csv_file, skiprows=1)
        with open(csv_file,'r') as f:
            self.picpath = f.readline().rstrip()
        self.transform = transform
        
    def __len__(self):
        """
        Return sample size
        """
        return self.csv_file.shape[0]
    
    def __getitem__(self, idx):
        """
        Get picture name, picture data and target of each sample
        
        Parameters:
        -----------
        idx: index of sample
        
        Returns:
        ---------
        picname: picture name
        picimg: picture data, save as a pillow instance
        condition: target of each sample (label)
        """
        # load pictures
        picname = np.array(self.csv_file['stimID'])
        condition = np.array(self.csv_file['condition'])
        picimg = Image.open(os.path.join(self.picpath, condition[idx], picname[idx]))
        if self.transform:
            picimg = self.transform(picimg)[None, ...]
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
            picimg = self.transform(picimg)[None, ...]
        return picname[idx], picimg, condition[idx]        

   
def load_brainimg(imgpath, ismask=False):
    """
    Load brain image identified by its suffix
    suffix now support
      
    Nifti: .nii.gz
    freesurfer: .mgz, .mgh
    cifti: .dscalar.nii, .dlabel.nii, .dtseries.nii
        
    Parameters:
    ------------
    imgpath: brain image data path
        
    Returns:
    ------------
    brain_img[np.array]: data of brain image
    header[header]: header of brain image
    """
    imgname = os.path.basename(imgpath)
    imgsuffix = imgname.split('.')[1:]
    imgsuffix = '.'.join(imgsuffix)

    if imgsuffix == 'nii.gz':
        brain_img = nib.load(imgpath).get_data()
        if not ismask:
            brain_img = np.transpose(brain_img,(3,0,1,2))
        header = nib.load(imgpath).header
    elif imgsuffix == 'mgz' or imgsuffix == 'mgh':
        brain_img = nib.freesurfer.load(imgpath).get_data()
        if not ismask:
            brain_img = np.transpose(brain_img, (3,0,1,2))
        header = nib.freesurfer.load(imgpath).header
    elif imgsuffix == 'dscalar.nii' or imgsuffix == 'dlabel.nii' or imgsuffix == 'dtseries.nii':
        brain_img, header = cifti.read(imgpath)
        if not ismask:
            brain_img = brain_img[...,None,None]
        else:
            brain_img = brain_img[...,None]
    else:
        raise Exception('Not support this format of brain image data, please contact with author to update this function.')
    return brain_img, header
    
    
def save_brainimg(imgpath, data, header):
    """
    Save brain image identified by its suffix
    suffix now support
     
    Nifti: .nii.gz
    freesurfer: .mgz, .mgh
    cifti: .dscalar.nii, .dlabel.nii, .dtseries.nii
        
    Parameters:
    ------------
    imgpath: brain image path to be saved
    data: brain image data matrix
    header: brain image header
        
    Returns:
    --------
    """
    imgname = os.path.basename(imgpath)
    imgsuffix = imgname.split('.')[1:]
    imgsuffix = '.'.join(imgsuffix)
    
    if imgsuffix == 'nii.gz':
        data = np.transpose(data,(1,2,3,0))
        outimg = nib.Nifti1Image(data, None, header)
        nib.save(outimg, imgpath)
    elif imgsuffix == 'mgz' or imgsuffix == 'mgh':
        data = np.transpose(data, (1,2,3,0))
        outimg = nib.MGHImage(data, None, header)
        nib.save(outimg, imgpath)
    elif imgsuffix == 'dscalar.nii' or imgsuffix == 'dlabel.nii' or imgsuffix == 'dtseries.nii':
        data = data[...,0,0]
        map_name = ['']*data.shape[0]
        bm_full = header[1]
        cifti.write(imgpath, data, (cifti.Scalar.from_names(map_names), bm_full))
    else:
        raise Exception('Not support this format of brain image data, please contact with author to update this function.')   
 
        
def save_activation(activation,outpath):
    """
    Save activaiton data as a csv file or mat format file to outpath
         csv format save a 2D.
            The first column is stimulus indexs
            The second column is channel indexs
            Each row is the activation of a filter for a picture
         mat format save a 2D or 4D array depend on the activation from convolution layer or fully connected layer.
            4D array Dimension:sitmulus x channel x pixel x pixel
            2D array Dimension:stimulus x activation
    Parameters:
    ------------
    activation[4darray]: sitmulus x channel x pixel x pixel
    outpath[str]:outpath and outfilename
    """
    imgname = os.path.basename(outpath)
    imgsuffix = imgname.split('.')[1:]
    imgsuffix = '.'.join(imgsuffix)

    if imgsuffix == 'csv':
        if len(activation.shape) == 4:
            activation2d = np.reshape(activation, (np.prod(activation.shape[0:2]), -1,), order='C')
            channelline = np.array([channel + 1 for channel in range(activation.shape[1])] * activation.shape[0])
            stimline = []
            for i in range(activation.shape[0]):
                a = [i + 1 for j in range(activation.shape[1])]
                stimline = stimline + a
            stimline = np.array(stimline)
            channelline = np.reshape(channelline, (channelline.shape[0], 1))
            stimline = np.reshape(stimline, (stimline.shape[0], 1))
            activation2d = np.concatenate((stimline, channelline, activation2d), axis=1)
        elif len(activation.shape) == 2:
            stim_indexs = np.arange(1, activation.shape[0] + 1)
            stim_indexs = np.reshape(stim_indexs, (-1, stim_indexs[0]))
            activation2d = np.concatenate((stim_indexs, activation), axis=1)
        np.savetxt(outpath, activation2d, delimiter=',')
    elif imgsuffix == 'mat':
        scipy.io.savemat(outpath,mdict={'activation':activation})
    else:
        raise Exception(
        'Not support this format of brain image data, please contact with author to update this function.')


class NetLoader:

    def __init__(self, net):
        """
        Load neural network model according to the net name.
        Meanwhile, hard code each network's information internally.

        Parameters:
        -----------
        net[str]: a neural network's name
        """
        if net == 'alexnet':
            self.model = torchvision.models.alexnet()
            self.model.load_state_dict(torch.load(os.path.join(DNNBRAIN_MODEL_DIR, 'alexnet_param.pth')))
            self.layer2indices = {'conv1': (0, 0), 'conv2': (0, 3), 'conv3': (0, 6), 'conv4': (0, 8),
                                  'conv5': (0, 10), 'fc1': (2, 1), 'fc2': (2, 4), 'fc3': (2, 6)}
            self.img_size = (224, 224)
        elif net == 'vgg11':
            self.model = torchvision.models.vgg11()
            self.model.load_state_dict(torch.load(os.path.join(DNNBRAIN_MODEL_DIR, 'vgg11_param.pth')))
            self.layer2indices = {'conv1': (0, 0), 'conv2': (0, 3), 'conv3': (0, 6), 'conv4': (0, 8),
                                  'conv5': (0, 11), 'conv6': (0, 13), 'conv7': (0, 16), 'conv8': (0, 18),
                                  'fc1': (2, 0), 'fc2': (2, 3), 'fc3': (2, 6)}
            self.img_size = (224, 224)
        else:
            raise Exception('Network was not supported, please contact author for implementation.')