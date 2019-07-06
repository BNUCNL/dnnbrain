try:
    import torch
    import torchvision
    import torchvision.models as models
    from torchvision import transforms, utils
except ModuleNotFoundError:
    raise Exception('Please install pytorch and torchvision in your work station')
    
import numpy as np

def dnn_activation(input,net,layer,channel=None):
    """
    Extract DNN activation
    
    Parameters:
    ------------
    input[dataset]: input image dataset
    net[str]: DNN network
    layer[int]: layer of DNN network, layer was counted from 1 (not 0)
    channel[int]: specify channel in layer of DNN network, channel was counted from 1 (not 0)
    
    Returns:
    ---------
    dnnact[numpy.array]: DNN activation, A 4D dataset with its format as pic*channel*unit*unit
    dnnpicname[numpy.array]: DNN picture name that correspond to each activation
    """
    if net == 'alexnet':
        model = models.alexnet(pretrained=True)
        conv = [1, 2, 3, 4, 5]                  # Number of convolution layers
        conv_rawlayer = [0, 3, 6, 8, 10]               # sequnence of convolution layers in net
    elif net == 'vgg11':
        model = models.vgg11(pretrained=True)
        conv = [1, 2, 3, 4, 5, 6, 7, 8]
        conv_rawlayer = [0, 3, 6, 8, 11, 13, 16, 18]
    else:
        raise Exception('Network was not supported, please contact author for implementation.')
    conv_map = dict(zip(conv,conv_rawlayer))
    
    assert layer in conv, 'Layer exceeds maximum values.'
    out_layer = conv_map[layer]
    actmodel_tmp = torch.nn.Sequential(*list(model.children())[0][0:out_layer+1])
    dnnact = []
    dnnpicname = []
    for _, picdata, target in input:
        dnnact_part = actmodel_tmp(picdata)
        dnnact.extend(dnnact_part.detach().numpy())
    dnnact = np.array(dnnact)
    if channel:
        dnnact = dnnact[:,channel-1,:,:]
    return dnnact