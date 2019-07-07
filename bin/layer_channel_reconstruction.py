#! /usr/bin/env python

"""
Use CNN activation to predict brain activation
Author: Yang Anmin
Reviewer:
"""

import os
import numpy as np
import argparse
import torch
from torch.nn import ReLU
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from dnnbrain.utils import iofiles


def main():
    parser = argparse.ArgumentParser(description='Visualize what a channel sees in the image')
    parser.add_argument('-net',
                            type = str,
                            required = True,
                            choices=['alexnet', 'vgg11'],
                            metavar = 'network name',
                            help = 'convolutional network name')
    paesr.add_argument('-input_path',
                            type = str,
                            required = True,
	                        metavar = 'picture path',
	                        help = 'picture path where pictures are stored'))
     parser.add_argument('-csv',
                        type = str,
                        required = True,
                        metavar = 'picture stimuli csv',
                        help = 'table contains picture names, conditions and picture onset time.\
                                This csv_file helps us connect cnn activation to brain images.\
                                Please organize your information as:\
                                ---------------------------------------\
                                stimID     condition   onset(optional) measurement(optional)\
                                face1.png  face        1.1             3\
                                face2.png  face        3.1             5\
                                scene1.png scene       5.1             4'
                        )
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
                            help = 'output directory where reconstructed pictures are stored')
    args = parser.parse_args()

    # select net
    netloader = iofiles.NetLoader(args.net)
    model = netloader.model

    picdataset = iofiles.PicDataset(args.csv, args.input_path, transform = None)
    for picname, picdata, _ in picdataset:
        out_image = layer_channel_reconstruction(model,picimg,args.layer,args.channel)
        im = Image.fromarray(out_image)
        outpath = args.output + '/' + picname + '_' + '%d' %args.layer + '_' + '%d' %args.channel
        im.save(outpath)
