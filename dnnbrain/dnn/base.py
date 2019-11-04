import cv2
import numpy as np
from PIL import Image

import os
from os.path import join as pjoin

import torch
from torchvision import transforms
from torchvision import models as torch_models
from dnnbrain.dnn import models as db_models



class ImageSet:
    """
    Build a dataset to load image
    """
    def __init__(self, img_dir, img_ids, labels=None, transform=None):
        """
        Initialize ImageSet

        Parameters:
        ----------
        img_dir[str]: images' parent directory
        img_ids[list]: Each img_id is a path which can find the image file relative to img_dir.
        labels[list]: Each image's label.
        transform[callable function]: optional transform to be applied on a stimulus.
        """
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.labels = np.ones(len(self.img_ids)) if labels is None else labels
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform

    def __len__(self):
        """
        Return the number of images
        """
        return len(self.img_ids)

    def __getitem__(self, indices):
        """
        Get image data and corresponding labels

        Parameter:
        ---------
        indices[int|list|slice]: subscript indices

        Returns:
        -------
        data[tensor]: image data with shape as (n_stim, n_chn, height, weight)
        labels[list]: image labels
        """
        # check availability and do preparation
        if isinstance(indices, int):
            tmp_ids = [self.img_ids[indices]]
            labels = [self.labels[indices]]
        elif isinstance(indices, list):
            tmp_ids = [self.img_ids[idx] for idx in indices]
            labels = [self.labels[idx] for idx in indices]
        elif isinstance(indices, slice):
            tmp_ids = self.img_ids[indices]
            labels = self.labels[indices]
        else:
            raise IndexError("only integer, slices (`:`) and list are valid indices")

        # load data
        data = torch.zeros(0)
        for img_id in tmp_ids:
            image = Image.open(pjoin(self.img_dir, img_id))  # load image
            image = self.transform(image)  # transform image
            image = torch.unsqueeze(image, 0)
            data = torch.cat((data, image))

        if data.shape[0] == 1:
            data = data[0]

        return data, labels


class VideoSet:
    """
    Dataset for video data
    """
    def __init__(self, vid_file, frame_nums, labels=None, transform=None):
        """
        Parameters:
        ----------
        vid_file[str]: video data file
        frame_nums[list]: sequence numbers of the frames of interest
        labels[list]: each frame's label
        transform[pytorch transform]
        """
        self.vid_cap = cv2.VideoCapture(vid_file)
        self.frame_nums = frame_nums
        self.labels = np.ones(len(self.frame_nums)) if labels is None else labels
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform

    def __getitem__(self, indices):
        """
        Get frame data and corresponding labels

        Parameter:
        ---------
        indices[int|list|slice]: subscript indices

        Returns:
        -------
        data[tensor]: frame data with shape as (n_stim, n_chn, height, weight)
        labels[list]: frame labels
        """
        # check availability and do preparation
        if isinstance(indices, int):
            tmp_nums = [self.frame_nums[indices]]
            labels = [self.labels[indices]]
        elif isinstance(indices, list):
            tmp_nums = [self.frame_nums[idx] for idx in indices]
            labels = [self.labels[idx] for idx in indices]
        elif isinstance(indices, slice):
            tmp_nums = self.frame_nums[indices]
            labels = self.labels[indices]
        else:
            raise IndexError("only integer, slices (`:`) and list are valid indices")

        # load data
        data = torch.zeros(0)
        for frame_num in tmp_nums:
            # get frame
            self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
            _, frame = self.vid_cap.read()
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame = self.transform(frame)  # transform frame
            frame = torch.unsqueeze(frame, 0)
            data = torch.cat((data, frame))

        if data.shape[0] == 1:
            data = data[0]

        return data, labels

    def __len__(self):
        """
        Return the number of frames
        """
        return len(self.frame_nums)
    


DNNBRAIN_MODEL = pjoin(os.environ['DNNBRAIN_DATA'], 'models')
class DNNLoader:
    """
    Load DNN model and initiate some information
    """

    def __init__(self, net=None):
        """
        Load neural network model

        Parameter:
        ---------
        net[str]: a neural network's name
        """
        self.model = None
        self.layer2loc = None
        self.img_size = None
        if net is not None:
            self.load(net)

    def load(self, net):
        """
        Load neural network model by net name

        Parameter:
        ---------
        net[str]: a neural network's name
        """
        if net == 'alexnet':
            self.model = torch_models.alexnet()
            self.model.load_state_dict(torch.load(
                pjoin(DNNBRAIN_MODEL, 'alexnet_param.pth')))
            self.layer2loc = {'conv1': ('features', '0'), 'conv1_relu': ('features', '1'),
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
            self.model = torch_models.vgg11()
            self.model.load_state_dict(torch.load(
                pjoin(DNNBRAIN_MODEL, 'vgg11_param.pth')))
            self.layer2loc = None
            self.img_size = (224, 224)
        elif net == 'vggface':
            self.model = db_models.Vgg_face()
            self.model.load_state_dict(torch.load(
                pjoin(DNNBRAIN_MODEL, 'vgg_face_dag.pth')))
            self.layer2loc = None
            self.img_size = (224, 224)
        else:
            raise ValueError("Not supported net name: {}, you can load model, "
                             "parameters, layer2loc, img_size manually by load_model".format(net))

    def load_model(self, model, parameters=None,
                   layer2loc=None, img_size=None):
        """
        Load DNN model, parameters, layer2loc and img_size manually

        Parameters:
        ----------
        model[nn.Modules]: DNN model
        parameters[state_dict]: Parameters of DNN model
        layer2loc[dict]: map layer name to its location in the DNN model
        img_size[tuple]: the input image size
        """
        self.model = model
        if parameters is not None:
            self.model.load_state_dict(parameters)
        self.layer2loc = layer2loc
        self.img_size = img_size

