try:
    import torch
    import torchvision
    from torch import nn
    import torchvision.models as models
    from torchvision import transforms, utils
except ModuleNotFoundError:
    raise Exception('Please install pytorch and torchvision in your work station')
    
import numpy as np

from dnnbrain.utils.iofiles import NetLoader


def dnn_activation(input, net, layer, channel=None):
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
    """
    loader = NetLoader(net)
    actmodel = truncate_net(loader.model, layer, loader.conv_indices)
    dnnact = []
    for _, picdata, target in input:
        dnnact_part = actmodel(picdata)
        # dnnact.append(torch.squeeze(dnnact_part, 0).detach().numpy())
        dnnact.extend(dnnact_part.detach().numpy())
    dnnact = np.array(dnnact)
    if channel:
        dnnact = dnnact[:, channel-1, :, :]
    return dnnact


def truncate_net(net, layer, conv_indices):
    """
    truncate the neural network at the specified layer number

    Parameters:
    -----------
    net[torch.nn.Module]: a neural network
    layer[int]: The sequence number of the layer which is connected to predict brain activity.
    conv_indices[list]: The convolution module's indices in the net

    Returns:
    --------
    truncated_net[torch.nn.Sequential]
    """
    conv_idx = conv_indices[layer-1]
    truncated_net = nn.Sequential(*list(net.children())[0][0:conv_idx+1])
    return truncated_net
