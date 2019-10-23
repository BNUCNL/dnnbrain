import os
import cv2
import scipy.io
import torch
import torchvision
import numpy as np

from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from dnnbrain.dnn.models import Vgg_face

DNNBRAIN_MODEL_DIR = os.environ['DNNBRAIN_MODEL_DIR']


class ImgDataset:
    """
    Build a dataset to load image
    """
    def __init__(self, par_path, img_ids, conditions=None, transform=None, crops=None):
        """
        Initialize ImgDataset

        Parameters:
        ------------
        par_path[str]: image parent path
        img_ids[sequence]: Each img_id is a path which can find the image file relative to par_path.
        conditions[sequence]: Each image's condition.
        transform[callable function]: optional transform to be applied on a sample.
        crops[array]: 2D array with shape (n_img, 4)
            Row index is corresponding to the index in img_ids.
            Each row is a bounding box which is used to crop the image.
            Each bounding box's four elements are:
                left_coord, upper_coord, right_coord, lower_coord.
        """
        self.par_path = par_path
        self.img_ids = img_ids
        self.conditions = np.ones(len(self.img_ids)) if conditions is None else conditions
        self.conditions_uniq = np.unique(self.conditions).tolist()
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        self.crops = crops

    def __len__(self):
        """
        Return sample size
        """
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Get image data and target label of each sample

        Parameters:
        -----------
        idx[int]: index of sample

        Returns:
        ---------
        image: image data
        label[int]: target of each sample (label)
        """
        # load image
        image = Image.open(os.path.join(self.par_path, self.img_ids[idx]))

        # crop image
        if self.crops is not None:
            image = image.crop(self.crops[idx])

        image = self.transform(image)  # transform image
        label = self.conditions_uniq.index(self.conditions[idx])  # get label
        return image, label


class VidDataset:
    """
    Dataset for video data
    """
    def __init__(self, vid_file, frame_nums, conditions=None, transform=None, crops=None):
        """
        Parameters:
        -----------
        vid_file[str]: video data file
        frame_nums[sequence]: sequence numbers of the frames of interest
        conditions[sequence]: each frame's condition
        transform[pytorch transform]
        crops[array]: 2D array with shape (n_img, 4)
            Row index is corresponding to the index in frame_nums.
            Each row is a bounding box which is used to crop the frame.
            Each bounding box's four elements are:
                left_coord, upper_coord, right_coord, lower_coord.
        """
        self.vid_cap = cv2.VideoCapture(vid_file)
        self.frame_nums = frame_nums
        self.conditions = np.ones(len(self.frame_nums)) if conditions is None else conditions
        self.conditions_uniq = np.unique(self.conditions).tolist()
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        self.crops = crops

    def __getitem__(self, idx):
        # get frame
        self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_nums[idx]-1)
        _, frame = self.vid_cap.read()
        frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # crop frame
        if self.crops is not None:
            frame_img = frame_img.crop(self.crops[idx])

        frame = self.transform(frame_img)  # transform frame
        trg_label = self.conditions_uniq.index(self.conditions[idx])  # get target
        return frame, trg_label

    def __len__(self):
        return len(self.frame_nums)


def read_imagefolder(parpath):
    """
    The function read from a already organized Image folder or a folder that only have images
    and return imgpath list and condition list
    for generate csv file more quickly.

    Parameters:
    ----------
    parpath[str]: parent path of images

    Return:
    ------
    imgpath[list]: contains all subpath of images in parpath
    condition[list]: contains categories of all images
    """
    test_set = list(os.walk(parpath))

    picpath = []
    condition = []
    if len(test_set) == 1:  # the folder only have images, the folder name will be the condition
        label = test_set[0]
        condition_name = os.path.basename(label[0])
        picpath_tem = label[2]
        condition_tem = [condition_name for i in label[2]]
        picpath.append(picpath_tem)
        condition.append(condition_tem)
    else:                   # the folder have have some sub-folders as pytorch ImageFolder,
        for label in test_set[1:]:
            condition_name = os.path.basename(label[0])
            picpath_tem = [condition_name + '/' + pic for pic in label[2]]
            condition_tem = [condition_name for i in label[2]]  # the sub-folders name will be the conditions.
            picpath.append(picpath_tem)
            condition.append(condition_tem)

    picpath = sum(picpath, [])
    condition = sum(condition, [])
    return picpath, condition


def save_activation(activation, outpath):
    """
    Save activaiton data as a csv file or mat format file to outpath
         csv format save a 2D.
            The first column is stimulus indexs
            The second column is channel indexs
            Each row is the activation of a filter for a image
         mat format save a 2D or 4D array depend on the activation from
             convolution layer or fully connected layer.
            4D array Dimension:sitmulus x channel x pixel x pixel
            2D array Dimension:stimulus x activation
    Parameters:
    ------------
    activation[4darray]: sitmulus x channel x pixel x pixel
    outpath[str]:outpath and outfilename
    """
    imgname = os.path.basename(outpath)
    imgsuffix = imgname.split('.')[-1]

    if imgsuffix == 'csv':
        if len(activation.shape) == 4:
            activation2d = np.reshape(
                    activation, (np.prod(activation.shape[0:2]), -1,),
                    order='C')
            channelline = np.array(
                    [channel + 1 for channel
                     in range(activation.shape[1])] * activation.shape[0])
            stimline = []
            for i in range(activation.shape[0]):
                a = [i + 1 for j in range(activation.shape[1])]
                stimline = stimline + a
            stimline = np.array(stimline)
            channelline = np.reshape(channelline, (channelline.shape[0], 1))
            stimline = np.reshape(stimline, (stimline.shape[0], 1))
            activation2d = np.concatenate(
                    (stimline, channelline, activation2d), axis=1)
        elif len(activation.shape) == 2:
            stim_indexs = np.arange(1, activation.shape[0] + 1)
            stim_indexs = np.reshape(stim_indexs, (-1, stim_indexs[0]))
            activation2d = np.concatenate((stim_indexs, activation), axis=1)
        np.savetxt(outpath, activation2d, delimiter=',')
    elif imgsuffix == 'mat':
        scipy.io.savemat(outpath, mdict={'activation': activation})
    else:
        np.save(outpath, activation)


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
                self.model.load_state_dict(torch.load(
                        os.path.join(DNNBRAIN_MODEL_DIR, 'alexnet_param.pth')))
                self.layer2indices = {'conv1': (0, 0), 'conv1_relu': (0, 1), 'conv1_maxpool': (0, 2), 'conv2': (0, 3),
                                      'conv2_relu': (0, 4), 'conv2_maxpool': (0, 5), 'conv3': (0, 6), 'conv3_relu': (0, 7),
                                      'conv4': (0, 8), 'conv4_relu': (0, 9),'conv5': (0, 10), 'conv5_relu': (0, 11),
                                      'conv5_maxpool': (0, 12), 'fc1': (2, 1), 'fc1_relu': (2, 2),
                                      'fc2': (2, 4), 'fc2_relu': (2, 5), 'fc3': (2, 6), 'prefc': (2,)}
                self.layer2keys = {'conv1': ('features', '0'), 'conv1_relu': ('features', '1'),
                                    'conv1_maxpool': ('features', '2'), 'conv2': ('features', '3'),
                                    'conv2_relu': ('features', '4'), 'conv2_maxpool': ('features', '5'),
                                    'conv3': ('features', '6'), 'conv3_relu': ('features', '7'),
                                    'conv4': ('features', '8'), 'conv4_relu': ('features', '9'),
                                    'conv5': ('features', '10'), 'conv5_relu': ('features', '11'),
                                    'conv5_maxpool': ('features', '12'), 'fc1': ('classifier', '1'),
                                    'fc1_relu': ('classifier', '2'), 'fc2': ('classifier', '4'),
                                    'fc2_relu': ('classifier', '5'), 'fc3': ('classifier', '6')}
                self.img_size = (224, 224)
            elif net == 'vgg11':
                self.model = torchvision.models.vgg11()
                self.model.load_state_dict(torch.load(
                        os.path.join(DNNBRAIN_MODEL_DIR, 'vgg11_param.pth')))
                self.layer2indices = {'conv1': (0, 0), 'conv2': (0, 3),
                                      'conv3': (0, 6), 'conv4': (0, 8),
                                      'conv5': (0, 11), 'conv6': (0, 13),
                                      'conv7': (0, 16), 'conv8': (0, 18),
                                      'fc1': (2, 0), 'fc2': (2, 3),
                                      'fc3': (2, 6), 'prefc':(2,)}
                self.img_size = (224, 224)
            elif net == 'vggface':
                self.model = Vgg_face()
                self.model.load_state_dict(torch.load(
                        os.path.join(DNNBRAIN_MODEL_DIR, 'vgg_face_dag.pth')))
                self.layer2indices = {'conv1': (0,), 'conv2': (2,),
                                      'conv3': (5,), 'conv4': (7,),
                                      'conv5': (10,), 'conv6': (12,),
                                      'conv7': (14,), 'conv8': (17,),
                                      'conv9': (19,), 'conv10': (21,),
                                      'conv11': (24,), 'conv12': (26,),
                                      'conv13': (28,), 'fc1': (31,),
                                      'fc2': (34,), 'fc3': (37,), 'prefc':(31,)}
                self.img_size = (224, 224)
        else:
            print('Not internal supported, please call netloader function'
                  'to assign model, layer2indices and image size.')
            self.model = None
            self.layer2indices = None
            self.img_size = None

    def load_model(self, dnn_model, model_param=None,
                   layer2indices=None, input_imgsize=None):
        """
        Load DNN model

        Parameters:
        -----------
        dnn_model[nn.Modules]: DNN model
        model_param[string/state_dict]: Parameters of DNN model
        layer2indices[dict]: Comparison table between layer name and
            DNN frame layer.
            Please make dictionary as following format:
                {'conv1': (0, 0), 'conv2': (0, 3), 'fc1': (2, 0)}
        input_imgsize[tuple]: the input image size
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


def read_dmask_csv(fpath):
    """
    Read pre-designed .dmask.csv file.

    Parameters:
    ----------
    fpath: path of .dmask.csv file

    Return:
    ------
    dmask_dict[OrderedDict]: Dictionary of the DNN mask information
    """
    # -load csv data-
    assert fpath.endswith('.dmask.csv'), 'File suffix must be .dmask.csv'
    with open(fpath) as rf:
        lines = rf.read().splitlines()

    # extract layers, channels and columns of interest
    dmask_dict = OrderedDict()
    for l_idx, line in enumerate(lines):
        if '=' in line:
            # layer
            layer, axes = line.split('=')
            dmask_dict[layer] = {'chn': None, 'col': None}

            # channels and columns
            axes = axes.split(',')
            while '' in axes:
                axes.remove('')
            assert len(axes) <= 2, \
                "The number of a layer's axes must be less than or equal to 2."
            for a_idx, axis in enumerate(axes, 1):
                assert axis in ('chn', 'col'), 'Axis must be from (chn, col).'
                numbers = [int(num) for num in lines[l_idx+a_idx].split(',')]
                dmask_dict[layer][axis] = numbers

    return dmask_dict


def save_dmask_csv(fpath, dmask_dict):
    """
    Generate .dmask.csv

    Parameters
    ---------
    fpath[str]: output file path, ending with .dmask.csv
    dmask_dict[dict]: Dictionary of the DNN mask information
    """
    assert fpath.endswith('.dmask.csv'), 'File suffix must be .dmask.csv'
    with open(fpath, 'w') as wf:
        for layer, axes_dict in dmask_dict.items():
            axes = []
            num_lines = []
            assert len(axes_dict) <= 2, \
                "The number of a layer's axes must be less than or equal to 2."
            for axis, numbers in axes_dict.items():
                assert axis in ('chn', 'col'), 'Axis must be from (chn, col).'
                if numbers is not None:
                    axes.append(axis)
                    num_line = ','.join(map(str, numbers))
                    num_lines.append(num_line)

            wf.write('{0}={1}\n'.format(layer, ','.join(axes)))
            for num_line in num_lines:
                wf.write(num_line+'\n')



def read_stim_csv(fpath):
    """
    Read pre-designed .stim.csv file.
    Parameters:
    -----------
    fpath[str]: Path of .stim.csv file.
        Format of .stim.csv of picture stimuli is
        --------------------------
        title=picture stimuli
        type=picture
        path=parent_dir_to_pictures
        [Several optional keys] (eg., hrf_tr=2)
        stim=stimID,[onset],[duration],[condition]
        meas=accuracy, reaction time
        pic1_path,0,1,cat,0.4,0.5
        pic2_path,1,1,dog,0.6,0.4
        pic3_path,2,1,cat,0.7,0.5
        ...,...,...,...,...,...
        Format of .stim.csv of video stimuli is
        --------------------------
        title=video stimuli
        type=video
        path=path_to_video_file
        [Several optional keys] (eg., hrf_tr=2)
        stim=stimID,[onset],[duration],[condition]
        meas=accuracy, reaction time
        1,0,1,cat,0.4,0.5
        2,1,1,dog,0.6,0.4
        3,2,1,cat,0.7,0.5
        ...,...,...,...,...,...
    Return:
    -------
    stim_dict[OrderedDict]: Dictionary of the stimuli information
    """
    # -load csv data-
    assert fpath.endswith('.stim.csv'), "File suffix must be .stim.csv"
    with open(fpath) as rf:
        lines = rf.read().splitlines()
    # remove null line
    while '' in lines:
        lines.remove('')
    stim_idx = [line.startswith('stim=') for line in lines].index(True)
    meas_idx = stim_idx + 1
    meta_lines = lines[:stim_idx]
    var_lines = lines[meas_idx+1:]

    # -handle csv data-
    # --operate meta_lines--
    stim_dict = {}
    for line in meta_lines:
        k, v = line.split('=')
        stim_dict[k] = v
    assert 'title' in stim_dict.keys(), "'title' needs to be included in meta data."
    assert 'type' in stim_dict.keys(), "'type' needs to be included in meta data."
    assert 'path' in stim_dict.keys(), "'path' needs to be included in meta data."

    # --operate var_lines--
    # prepare keys
    stim_keys = lines[stim_idx].split('=')[1].split(',')
    assert 'stimID' in stim_keys, "'stimID' must be included in 'stim=' line."
    n_stim_key = len(stim_keys)
    meas_keys = lines[meas_idx].split('=')[1].split(',')
    while '' in meas_keys:
        meas_keys.remove('')

    # prepare variable data
    var_data = [line.split(',') for line in var_lines]
    var_data = list(zip(*var_data))
    if stim_dict['type'] == 'picture':
        # data type for stimID or condition is str, others float.
        for i, v in enumerate(var_data[:n_stim_key]):
            dtype = np.str if stim_keys[i] in ['stimID', 'condition'] else np.float
            var_data[i] = np.asarray(v, dtype=dtype)
    elif stim_dict['type'] == 'video':
        for i, v in enumerate(var_data[:n_stim_key]):
            if stim_keys[i] == 'stimID':
                dtype = np.int
            elif stim_keys[i] == 'condition':
                dtype = np.str
            else:
                dtype = np.float
            var_data[i] = np.asarray(v, dtype=dtype)
    else:
        raise ValueError('not supported stimulus type: {}'.format(stim_dict['type']))

    # get stimulus variable dict
    stim_var_dict = OrderedDict()
    for idx, key in enumerate(stim_keys):
        stim_var_dict[key] = var_data[idx]
    stim_dict['stim'] = stim_var_dict

    # get measurement variable dict
    if meas_keys:
        for i, v in enumerate(var_data[n_stim_key:], n_stim_key):
            var_data[i] = np.asarray(v, dtype=np.float)
        meas_var_dict = OrderedDict()
        for idx, key in enumerate(meas_keys, n_stim_key):
            meas_var_dict[key] = var_data[idx]
        stim_dict['meas'] = meas_var_dict
    else:
        stim_dict['meas'] = None

    return stim_dict

def save_stim_csv(fpath, title, type, path, stim_var_dict,
                  meas_var_dict=None, opt_meta=None):
    """
    Generate .stim.csv
    Parameters:
    ------------
    fpath[str]: output file path, ending with .stim.csv
    title[str]: customized title
    type[str]: stimulus type in ('picture', 'video')
    path[str]: path_to_stimuli
        If type is 'picture', the path is the parent directory of the pictures.
        If type is 'video', the path is the file path of the video.
    stim_var_dict[dict]: dictionary of stimulus variables
    meas_var_dict[dcit]: dictionary of measurement variables
    opt_meta[dict]: some other optional meta data
    """
    assert fpath.endswith('.stim.csv'), "File suffix must be .stim.csv"
    with open(fpath, 'w') as wf:
        # write the tile, type and path
        wf.write('title={}\n'.format(title))
        wf.write('type={}\n'.format(type))
        wf.write('path={}\n'.format(path))

        # write optional meta data
        if opt_meta is not None:
            for k, v in opt_meta.items():
                wf.write('{0}={1}\n'.format(k, v))

        # write stim and meas
        wf.write('stim={}\n'.format(','.join(stim_var_dict.keys())))
        var_data = np.array(list(stim_var_dict.values()), dtype=np.str).T
        if meas_var_dict is None:
            wf.write('meas=\n')
        else:
            wf.write('meas={}\n'.format(','.join(meas_var_dict.keys())))
            var_data = np.c_[var_data, np.array(list(meas_var_dict.values()), dtype=np.str).T]

        var_data = [','.join(row) for row in var_data]
        wf.write('\n'.join(var_data))