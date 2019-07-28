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
                        stimID          condition   onset(optional) measurement(optional)
                        download/face1  face        1.1             3
                        mgh/face2.png   face        3.1             5
                        scene1.png      scene       5.1             4
        
        transform[callable function]: optional transform to be applied on a sample.
        """
        self.csv_file = pd.read_csv(csv_file, skiprows=1)
        with open(csv_file,'r') as f:
            self.picpath = f.readline().rstrip()
        self.transform = transform
        picname = np.array(self.csv_file['stimID'])
        condition = np.array(self.csv_file['condition'])
        self.picname = picname
        self.condition = condition
        
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
        target_label: target of each sample (label)
        """
        # load pictures
        target_name = np.unique(self.condition)
        picimg = Image.open(os.path.join(self.picpath, self.picname[idx])).convert('RGB')
        target_label = target_name.tolist().index(self.condition[idx])
        if self.transform:
            picimg = self.transform(picimg)
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
            picimg = self.transform(picimg)
        return picimg, target_label
        
    def get_picname(self, idx):
        """
        Get picture name and its condition (target condition)
        
        Parameters:
        -----------
        idx: index of sample
        
        Returns:
        ---------
        picname: picture name
        condition: target condition
        """
        return os.path.basename(self.picname[idx]), self.condition[idx]


def generate_stim_csv(parpath, picname_list, condition_list, outpath, onset_list=None, behavior_measure=None):
    """
    Automatically generate stimuli table file.
    Noted that the stimuli table file satisfied follwing structure and sequence needs to be consistent:
    
    [PICDIR]
    stimID              condition   onset(optional) measurement(optional)
    download/face1.png  face        1.1             3
    mgh/face2.png       face        3.1             5
    scene1.png          scene       5.1             4
    
    Parameters:
    ------------
    parpath[str]: parent path contains stimuli pictures
    picname_list[list]: picture name list, each element is a relative path (string) of a picture
    condition_list[list]: condition list
    outpath[str]: output path
    onset_list[list]: onset time list
    behavior_measure[dictionary]: behavior measurement dictionary
    """    
    assert len(picname_list) == len(condition_list), 'length of picture name list must be equal to condition list.'
    assert os.path.basename(outpath).endswith('csv'), 'Suffix of outpath should be .csv'
    picnum = len(picname_list)
    if onset_list is not None:
        onset_list = [str(ol) for ol in onset_list]
    if behavior_measure is not None:
        list_int2str = lambda v: [str(i) for i in v]
        behavior_measure = {k:list_int2str(v) for k, v in behavior_measure.items()}
    with open(outpath, 'w') as f:
        # First line, parent path
        f.write(parpath+'\n')
        # Second line, key names
        table_keys = 'stimID,condition'
        if onset_list is not None:
            table_keys += ','
            table_keys += 'onset'
        if behavior_measure is not None:
            table_keys += ','
            table_keys += ','.join(behavior_measure.keys())
        f.write(table_keys+'\n')
        # Three+ lines, Data
        for i in range(picnum):
            data = picname_list[i]+','+condition_list[i]
            if onset_list is not None:
                data += ','
                data += onset_list[i]
            if behavior_measure is not None:
                for bm_keys in behavior_measure.keys():
                    data += ','
                    data += behavior_measure[bm_keys][i]
            f.write(data+'\n')
            
        
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
        np.save(outpath,activation)


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