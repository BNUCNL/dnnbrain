try:
    import torch
    import torchvision
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
    out_layer = loader.conv_indices[layer-1]
    actmodel = torch.nn.Sequential(*list(loader.model.children())[0][0:out_layer+1])
    dnnact = []
    for _, picdata, target in input:
        dnnact_part = actmodel(picdata)
        # dnnact.append(torch.squeeze(dnnact_part, 0).detach().numpy())
        dnnact.extend(dnnact_part.detach().numpy())
    dnnact = np.array(dnnact)
    if channel:
        dnnact = dnnact[:, channel-1, :, :]
    return dnnact, 1
