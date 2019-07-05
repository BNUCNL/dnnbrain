"""
visualize what a channel of a layer sees in the image
Author: Yang Anmin @ BNU
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


def rescale_grads(map,gradtype="all"):
    map = map - map.min()
    map /= map.max()
    return map


def lyaer_channel_visualization(net,input,layer,channel,output):
    """
    Visualize DNN layer_channel

    Parameters:
    ------------
    net[str]: DNN network
    input[str]: input path
    layer[int]: layer number
    channel[int]: channel number
    out[str]: output path
    
    """
    img = Image.open(input)
    transfromation = transforms.Compose([transforms.ToTensor()])
    img = transfromation(img)
    img = img[np.newaxis]

    if args.net == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        out = model(img)
        _, preds = torch.max(out.data, 1)
        target_class=preds.numpy()[0]
    elif args.net == 'vgg11':
        model = torchvision.models.vgg11(pretrained=True)
        out = model(img)
        _, preds = torch.max(out.data, 1)
        target_class=preds.numpy()[0]
    else:
        raise Exception('Network was not supported, please contact author for implementation.')



    img.requires_grad=True
    GBP = GuidedBackprop(model)
    guided_grads = GBP.generate_gradients(img, target_class, cnn_layer, channel)
    all_sal = rescale_grads(guided_grads,gradtype="all")
    out_image = torch.from_numpy(all_sal).permute(1,2,0)
    out_image = out_image.numpy()
    #img_name ='%d' %n_picture + '_' + 'layer' + '_' + 'channel'
    save_image(out_image, output)

if __name__ == '__main__':
    lyaer_channel_visualization()
