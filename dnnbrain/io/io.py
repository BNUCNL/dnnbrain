import os
import torch

from os.path import join as pjoin
from torchvision import models as torch_models
from dnnbrain.dnn import models as db_models

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
