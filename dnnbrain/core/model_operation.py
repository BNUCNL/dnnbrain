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
    actmodel = truncate_net(loader.model, loader.layer2indices[layer])
    dnnact = []
    for _, picdata, target in input:
        dnnact_part = actmodel(picdata)
        dnnact.extend(dnnact_part.detach().numpy())
    dnnact = np.array(dnnact)
    if channel:
        channel_new = [cl-1 for cl in channel]
        dnnact = dnnact[:, tuple(channel_new), :, :]
    return dnnact


def truncate_net(net, indices):
    """
    truncate the neural network at the specified layer number

    Parameters:
    -----------
    net[torch.nn.Module]: a neural network
    indices[iterator]: a sequence of raw indices to find a layer

    Returns:
    --------
    truncated_net[torch.nn.Sequential]
    """
    if len(indices) > 1:
        tmp = list(net.children())[:indices[0]]
        next = list(net.children())[indices[0]]
        return nn.Sequential(*(tmp+[truncate_net(next, indices[1:])]))
    elif len(indices) == 1:
        return nn.Sequential(*list(net.children())[:indices[0]+1])
    else:
        raise ValueError("Layer indices must not be empty!")
