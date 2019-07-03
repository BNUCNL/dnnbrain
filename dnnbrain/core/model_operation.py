import torch
import torchvision
import torchvision.models as models
from torchvision import transforms, utils
import numpy as np

def dnn_activation(input,net,layer,filter=None):


    assert net in ['alexnet','vgg11'], "please specify a net"

    if net == 'alexnet':
        model = models.alexnet(pretrained=True)
        conv = [1, 2, 3, 4, 5]                  # Number of convolution layers
        conv_realayer = [0,3,6,8,10]               # sequnence of convolution layers in net
    elif net == 'vgg11':
        model = models.vgg11(pretrained=True)
        conv = [1, 2, 3, 4, 5, 6, 7, 8]
        conv_realayer = [0, 3, 6, 8, 11, 13, 16, 18]
    conv_map = dict(zip(conv,conv_realayer))

    assert layer in conv, 'please specify a layer or your layer beyond the net!'

    out_layer = conv_map[layer]
    new_model = torch.nn.Sequential(*list(model.children())[0][0:out_layer+1])
    activation = new_model(input)
    activation = activation.detach().numpy()            # return a 4D shape ndarray （stim filter pixel pixel）
    if filter:
        activation = activation[:,filter-1,:,:]
    return activation, new_model


if __name__ == '__main__':
    input = torch.rand(8,3,224,224)
    net = 'alexnet'
    layer = 2
    filter = 1
    activation,model= dnn_activation(input,net,layer,filter)
    print(activation.shape)
    print(model)