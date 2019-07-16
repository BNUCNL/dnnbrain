try:
    import torch
    import torchvision
    from torch import nn
    import torchvision.models as models
    from torchvision import transforms, utils
except ModuleNotFoundError:
    raise Exception('Please install pytorch and torchvision in your work station')
    
import numpy as np

from dnnbrain.utils import iofiles


def dnn_activation(input, net, layer, channel=None):
    """
    Extract DNN activation
    
    Parameters:
    ------------
    input[dataset]: input image dataset
    net[str]: DNN network
    layer[str]: layer name of a DNN network
    channel[list]: specify channel in layer of DNN network, channel was counted from 1 (not 0)
    
    Returns:
    ---------
    dnnact[numpy.array]: DNN activation, A 4D dataset with its format as pic*channel*unit*unit
    """
    loader = iofiles.NetLoader(net)
    if 'conv' in layer:
        dnnact =  dnn_activaiton_conv(input,loader,layer)
    elif 'fc' in layer:
        dnnact = dnn_activation_fc(input,loader,layer)
    else:
        raise Exception("Not support this layer,please entry conv or fc")
    if channel:
        channel_new = [cl - 1 for cl in channel]
        dnnact = dnnact[:, channel_new, :, :]
    return dnnact


def dnn_activaiton_conv(input,netloader,layer):
    actmodel = truncate_net_conv(netloader.model, netloader.layer2indices[layer])
    dnnact = []
    for _, picdata, target in input:
        dnnact_part = actmodel(picdata)
        dnnact.extend(dnnact_part.detach().numpy())
    dnnact = np.array(dnnact)
    return dnnact


def dnn_activation_fc(input,netloader,layer):
    actmodel = netloader.model
    indices = netloader.layer2indices[layer]
    new_classifier = nn.Sequential(*list(netloader.model.children())[-1][:indices[1] + 1])
    if hasattr(actmodel,'classifier'):
        actmodel.classifier = new_classifier
    elif hasattr(actmodel,'fc'):
        actmodel.fc = new_classifier
    dnnact = []
    for _, picdata, target in input:
        dnnact_part = actmodel(picdata)
        dnnact.extend(dnnact_part.detach().numpy())
    dnnact = np.array(dnnact)
    return dnnact


def truncate_net_conv(net, indices):
    """
    truncate the neural network at the specified convolution layer

    Parameters:
    -----------
    net[torch.nn.Module]: a neural network
    indices[iterator]: a sequence of raw indices to find a layer

    Returns:
    --------
    truncated_net[torch.nn.Sequential]
    """
    if len(indices) > 1:
        tmp  = list(net.children())[:indices[0]]
        next = list(net.children())[indices[0]]
        return nn.Sequential(*(tmp+[truncate_net_conv(next, indices[1:])]))
    elif len(indices) == 1:
        return nn.Sequential(*list(net.children())[:indices[0]+1])
    else:
        raise ValueError("Layer indices must not be empty!")


def dnn_finetuning(netloader,layer,out_features):
    """can reconstruct from any layer"""
    dnn_model = netloader.model
    indices = netloader.layer2indices[layer]
    old_classifier = list(netloader.model.children())[-1][:indices[1]]
    in_features = list(netloader.model.children())[-1][indices[1]].in_features
    new_classifier =   nn.Sequential(*(old_classifier + list(nn.Linear(in_features,out_features))))

    if hasattr(dnn_model,'classifier'):
        dnn_model.classifier = new_classifier
    elif hasattr(dnn_model,'fc'):
        dnn_model.fc = new_classifier
    return dnn_model