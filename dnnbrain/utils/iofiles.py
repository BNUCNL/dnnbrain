import os
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
    def __init__(self, csv_file, picpath, transform=None):
        """
        Initialize PicDataset
        
        Parameters:
        ------------
        csv_file[str/pd.DataFrame]: table contains picture names, conditions and picture onset time.
                                     This csv_file helps us connect cnn activation to brain images.
                                     Please organize your information as:
                                     ---------------------------------------
                                     stimID     condition   onset(optional) measurement(optional)
                                     face1.png  face        1.1             3
                                     face2.png  face        3.1             5
                                     scene1.png scene       5.1             4
        picpath[str]: parent path of pictures.
        transform[callable function]: optional transform to be applied on a sample.
        """
        if isinstance(csv_file,str):
            csv_file = pd.read_csv(csv_file)
        self.csv_file = csv_file
        self.picpath = picpath
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
        return picname[idx], picimg, condition[idx]        

   
def load_brainimg(imgpath):
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
    """
    imgname = os.path.basename(imgpath)
    imgsuffix = imgname.split('.')[1:]
    imgsuffix = '.'.join(imgsuffix)

    if imgsuffix == 'nii.gz':
        brain_img = nib.load(imgpath).get_data()
        header = nib.load(imgpath).header
    elif imgsuffix == 'mgz' or imgsuffix == 'mgh':
        brain_img = nib.freesurfer.load(imgpath).get_data()
        header = nib.freesurfer.load(imgpath).header
            
    elif imgsuffix == 'dscalar.nii' or imgsuffix == 'dlabel.nii' or imgsuffix == 'dtseries.nii':
        brain_img, header = cifti.read(imgpath)
    else:
        raise Exception('Not support this format of brain image data, please contact with author to update this function.')
    assert brain_img.ndim == 4, "Please reconstruct your image data as an 4D image with a dimension for picture."
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
        outimg = nib.Nifti1Image(data, None, header)
        nib.save(outimg, imgpath)
    elif imgsuffix == 'mgz' or imgsuffix == 'mgh':
        outimg = nib.MGHImage(outimg, None, header)
        nib.save(outimg, imgpath)
    elif imgsuffix == 'dscalar.nii' or imgsuffix == 'dlabel.nii' or imgsuffix == 'dtseries.nii':
        map_name = ['']*outimg.shape[0]
        bm_full = header[1]
        cifti.write(imgpath, outimg, (cifti.Scalar.from_names(map_names), bm_full))
    else:
        raise Exception('Not support this format of brain image data, please contact with author to update this function.')   
 
        
def save_activation_to_csv(activation,outpath,net,layer,channel=None):
    """
    Save activaiton data to a csv file in outpath

    Parameters:
    ------------
    activation[4darray]: sitmulus x channel x pixel x pixel
    outpath[str]:outpath and outfilename
    net[str]:neuron network name
    layer[int]:the number of layer of network
    channel[list]: the number list of channel/filter
    """
    activation2d = np.reshape(activation,(np.prod(activation.shape[0:2]),-1,),order='C')
    channelline = np.array([channel+1 for channel in range(activation.shape[1])]*activation.shape[0])
    stimline = []
    for i in range(activation.shape[0]):
        a = [i + 1 for j in range(activation.shape[1])]
        stimline = stimline + a
    stimline = np.array(stimline)
    channelline = np.reshape(channelline, (channelline.shape[0], 1))
    stimline = np.reshape(stimline, (stimline.shape[0], 1))
    activation2d = np.concatenate((stimline,channelline,activation2d),axis=1)
    if channel is None:
        np.savetxt('{}_{}_{}.csv'.format(outpath,net,layer),activation2d,delimiter = ',')
    else:
        np.savetxt('{}{}_{}_{}.csv'.format(outpath, net,layer,channel), activation2d, delimiter=',')
        

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
            self.conv_indices = [0, 3, 6, 8, 10]  # map convolution layer number to raw index in model
            self.img_size = (224, 224)
        elif net == 'vgg11':
            self.model = torchvision.models.vgg11()
            self.model.load_state_dict(torch.load(os.path.join(DNNBRAIN_MODEL_DIR, 'vgg11_param.pth')))
            self.conv_indices = [0, 3, 6, 8, 11, 13, 16, 18]
            self.img_size = (224, 224)
        else:
            raise Exception('Network was not supported, please contact author for implementation.')
