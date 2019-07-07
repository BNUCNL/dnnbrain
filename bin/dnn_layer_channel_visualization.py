#! /usr/bin/env python

"""
Use CNN activation to predict brain activation
Author: Yang Anmin
Reviewer:
"""
from dnnbrain.utils import iofiles

import numpy as np
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append('/home/dell/Desktop/src') # locate layer_activation_with_guided_backprop

try:
    from layer_activation_with_guided_backprop import GuidedBackprop
    from misc_functions import save_image
except ModuleNotFoundError:
    raise Exception('Please install layer_activation_with_guided_backprop in your work station.')

def main():
    parser = argparse.ArgumentParser(description='Visualize what a channel sees in the image')
    parser.add_argument('-net',
                        type = str,
                        required = True,
                        metavar = 'CNN',
                        help = 'convolutional network name')
    parser.add_argument('-input',
                        type = str,
                        required = True,
                        metavar = 'Input Image Directory',
                        help = 'image path')
    parser.add_argument('-layer',
                        type = int,
                        required = True,
                        metavar = 'Layer',
                        help = 'layer of the channel(s)')
    parser.add_argument('-channel',
                        type = int,
                        required = True,
                        metavar = 'Channel',
                        help = 'channel of interest')
    parser.add_argument('-output',
                        type = str,
                        required = True,
                        metavar = 'Output Directory',
                        help = 'output directory')
    args = parser.parse_args()


if __name__ == '__main__':
    main()
