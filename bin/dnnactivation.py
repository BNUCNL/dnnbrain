import torch
import torchvision
import torchvision.models as models
from torchvision import transforms, utils
import numpy as np
from dnnbrain.dnnbrain.utils import iofiles as io
from dnnbrain.dnnbrain.core.model_operation import dnn_activation

import argparse

"""
To extract layers activation of CNN.

Author: Yukun Qu @ BNU
Reviewer:
"""




def main():
    parser = argparse.ArgumentParser(description='Use CNN activation to predict brain activation')
    parser.add_argument('-net',
                        type = str,
                        required = True,
						metavar = 'cnn_name',
                        help = 'convolutional network name')
    parser.add_argument('-in',
                        type = str,
                        required = True,
						metavar = 'Input Directory',
                        help = 'stimuli path')
    parser.add_argument('-layer',
                        type = int,
                        required = True,
						metavar = 'Layer of cnn',
                        help = 'number of layer')
    parser.add_argument('-channel',
                        type = int,
                        required = False,
						metavar = 'Channel',
                        help = 'activation for specific channels/filter')
    parser.add_argument('-out',
                        type = str,
                        required = True,
						metavar = 'Output Directory',
                        help = 'output directory')
    args = parser.parse_args()

# get img_tensor


def save_activation():
    pass
