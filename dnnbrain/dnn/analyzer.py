try:
    import torch
    import torchvision
    from torch import nn
    from torchvision import models
    from torchvision import transforms, utils
except ModuleNotFoundError:
    raise Exception('Please install pytorch and torchvision in your work station')
    
import numpy as np
from dnnbrain.dnn.models import dnn_truncate
from dnnbrain.dnn import io as iofiles

def dnn_activation(input, netname, layer, channel=None):
    """
    Extract DNN activation

    Parameters:
    ------------
    input[dataloader]: input image dataloader
    netname[str]: DNN network
    layer[str]: layer name of a DNN network
    channel[list]: specify channel in layer of DNN network, channel was counted from 1 (not 0)

    Returns:
    ---------
    dnnact[numpy.array]: DNN activation, A 4D dataset with its format as pic*channel*unit*unit
    """
    loader = iofiles.NetLoader(netname)
    actmodel = dnn_truncate(loader, layer)
    actmodel.eval()
    dnnact = []
    for picdata, target in input:
        dnnact_part = actmodel(picdata)
        dnnact.extend(dnnact_part.detach().numpy())
    dnnact = np.array(dnnact)

    if channel:
        channel_new = [cl - 1 for cl in channel]
        dnnact = dnnact[:, channel_new, :, :]
    return dnnact