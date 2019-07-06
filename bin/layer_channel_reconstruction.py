#! /usr/bin/env python

"""
Use CNN activation to predict brain activation
Author: Yang Anmin
Reviewer:
"""

import numpy as np
import argparse
import torch
from torch.nn import ReLU
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Visualize what a channel sees in the image')
    parser.add_argument('-net',
                        type = str,
                        required = True,
                        choices=['alexnet', 'vgg11']
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


    #  select net
    if args.net == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
    elif args.net == 'vgg11':
        model = torchvision.models.vgg11(pretrained=True)
    else:
        raise Exception('Network was not supported, please contact author for implementation.')

    # picture reconstruction
    lyaer_channel_reconstruction(args.net,args.input,args.layer,args.channel)

    # save Image
    def save_image(im, path):
        """
            Saves a numpy matrix or PIL image as an image
        Args:
            im_as_arr (Numpy array): Matrix of shape DxWxH
            path (str): Path to the image
        """
        if isinstance(im, (np.ndarray, np.generic)):
            im = format_np_output(im)
            im = Image.fromarray(im)
        im.save(path)
    save_image(out_image, args.output)
