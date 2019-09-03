import os
import scipy.io
import numpy as np
import pandas as pd
from dnnbrain.dnn.models import Vgg_face

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
    def __init__(self, csv_file, transform=None, crop=None):
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
        crop[bool]:crop picture optionally by a bounding box.
                   The coordinates of bounding box for crop pictures should be measurements in csv_file.
                   The label of coordinates in csv_file should be left_coord,upper_coord,right_coord,lower_coord.
        """
        self.csv_file = pd.read_csv(csv_file, skiprows=1)
        with open(csv_file,'r') as f:
            self.picpath = f.readline().rstrip()
        self.transform = transform
        picname = np.array(self.csv_file['stimID'])
        condition = np.array(self.csv_file['condition'])
        self.picname = picname
        self.condition = condition
        self.crop = crop
        if self.crop:
            self.left = np.array(self.csv_file['left_coord'])
            self.upper = np.array(self.csv_file['upper_coord'])
            self.right = np.array(self.csv_file['right_coord'])
            self.lower = np.array(self.csv_file['lower_coord'])

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
        if self.crop:
            picimg = picimg.crop((self.left[idx],self.upper[idx],self.right[idx],self.lower[idx]))
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


def read_Imagefolder(parpath):
    """
    Get picture path and conditions of a Imagefolder directory

    Parameters:
    ------------
    parpath[str]: Parent path of ImageFolder.
    
    Returns:
    ---------
    picpath[list]: picture path list
    conditions[list]: condition list
    """
    targets = os.listdir(parpath)
    picname_tmp = [os.listdir(os.path.join(parpath, tg)) for tg in targets]
    picnames = [pn for sublist in picname_tmp for pn in sublist]
    conditions = [tg for tg in targets for _ in picname_tmp]
    picpath = [os.path.join(conditions[i], picnames[i]) for i, _ in enumerate(picnames)]
    return picpath, conditions


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
    def __init__(self, net=None):
        """
        Load neural network model

        Parameters:
        -----------
        net[str]: a neural network's name
        """
        netlist = ['alexnet', 'vgg11', 'vggface']
        if net in netlist:
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
            elif net == 'vggface':
                self.model = Vgg_face()
                self.model.load_state_dict(torch.load(os.path.join(DNNBRAIN_MODEL_DIR, 'vgg_face_dag.pth')))
                self.layer2indices = {'conv1':(0,),'conv2':(2,),'conv3':(5,),'conv4':(7,),'conv5':(10,),'conv6':(12,),
                                      'conv7':(14,),'conv8':(17,),'conv9':(19,),'conv10':(21,),'conv11':(24,),
                                      'conv12':(26,),'conv13':(28,),'fc1':(31,),'fc2':(34,),'fc3':(37,)}
                self.img_size = (224, 224)
        else:
            print('Not internal supported, please call netloader function to assign model, layer2indices and picture size.')
            self.model = None
            self.layer2indices = None
            self.img_size = None
    
    def load_model(self, dnn_model, model_param = None, layer2indices = None, input_imgsize = None):
        """
        Load DNN model
        
        Parameters:
        -----------
        dnn_model[nn.Modules]: DNN model
        model_param[string/state_dict]: Parameters of DNN model
        layer2indices[dict]: Comparison table between layer name and DNN frame layer.
                             Please make dictionary as following format:
                             {'conv1': (0, 0), 'conv2': (0, 3), 'fc1': (2, 0)}
        input_imgsize[tuple]: the input picture size
        """
        self.model = dnn_model
        if model_param is not None:
            if isinstance(model_param, str):
                self.model.load_state_dict(torch.load(model_param))
            else:
                self.model.load_state_dict(model_param)
        self.layer2indices = layer2indices
        self.img_size = input_imgsize
        print('You had assigned a model into netloader.')
        
        
    def read_dmask_csv(dmask_file):
        'read the dnn mask csv file'
        with open(dmask_file,'r') as f:
            meta_data = [x.rstrip() for i, x in enumerate(f) if i<=2]
            resp_type = meta_data[0]
            tr = np.float(meta_data[1])
        
        resp_data = pd.read_csv(dmask_file, skiprows=2)
        return resp_type, tr, list(resp_data.keys()), resp_data.values
