"""
visualize what a channel of a layer sees in the image
Author: Yang Anmin @ BNU
Reviewer:

"""

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
sys.path.append('...') # find GuidedBackprop
from PIL import Image
from layer_activation_with_guided_backprop import GuidedBackprop



def lyaer_channel_visualization():
    parser = argparse.ArgumentParser(description='Visualize what a channel sees in the image')
    parser.add_argument('-net',
                        type = str,
                        required = True,
                        metavar = 'CNN',
                        help = 'convolutional network name')
    parser.add_argument('-in',
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
    parser.add_argument('-out',
                        type = str,
                        required = True,
                        metavar = 'Output Directory',
                        help = 'output directory')
    args = parser.parse_args()



    def rescale_grads(map,gradtype="all"):
        map = map - map.min()
        map /= map.max()
        return map

    input_image = Image.open(args.in)
    img = input_image[np.newaxis]
    cnn_layer = args.layer
    channel = args.channel
    path = args.out

    if args.net == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        out = model_alexnet(img)
        _, preds = torch.max(out.data, 1)
        target_class=preds.numpy()[0]
    elif args.net == 'vgg11':
        model = torchvision.models.vgg11(pretrained=True)
        out = model_vgg(img)
        _, preds = torch.max(out.data, 1)
        target_class=preds.numpy()[0]
    else:
        raise Exception('Network was not supported, please contact author for implementation.')



    img.requires_grad=True
    GBP = GuidedBackprop(model)
    guided_grads = GBP.generate_gradients(img, target_class, cnn_layer, channel)
    all_sal = rescale_grads(guided_grads,gradtype="all")
    out_image = torch.from_numpy(all_sal).permute(1,2,0)
    img_name ='%d' %n_picture + '_' + 'cnn_layer' + '_' + 'channel'
    plt.savefig(img_name, path)

 if __name__ == '__main__':
     lyaer_channel_visualization():
