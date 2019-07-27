from torch import nn


def dnn_truncate(net, indices, layer):
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
    if 'conv' in layer:
        if len(indices) > 1:
            tmp  = list(net.children())[:indices[0]]
            next = list(net.children())[indices[0]]
            actmodel = nn.Sequential(*(tmp+[dnn_truncate(next, indices[1:],layer='conv')]))
        elif len(indices) == 1:
            actmodel = nn.Sequential(*list(net.children())[:indices[0]+1])
        else:
            raise ValueError("The network has no this layer.")
    elif 'fc' in layer:
        actmodel = net
        new_classifier = nn.Sequential(*list(net.children())[-1][:indices[1] + 1])
        if hasattr(actmodel, 'classifier'):
            actmodel.classifier = new_classifier
        elif hasattr(actmodel, 'fc'):
            actmodel.fc = new_classifier
        else:
            raise Exception("The network has no this layer.")
    else:
        raise Exception("The input of layer format is not right,input just like conv5 or fc1")

    return actmodel


class DNN2BrainNet(nn.Module):

    def __init__(self, truncated_net, channel_unit_num, fc_out_num, channel=None):
        """
        Connect the truncated_net to a full connection layer.

        Parameters:
        -----------
        truncated_net[torch.nn.Module]: a truncated neural network from the pretrained network
        channel_unit_num[int]: the number of units of each channel of the last layer in the truncated network
        fc_out_num[int]: the number of the out channels of the full connection layer
        channel[iterator]: The sequence numbers of out channels of the selected layer.
        """
        super(DNN2BrainNet, self).__init__()
        self.truncated_net = truncated_net
        if channel is None:
            channel_num = list(self.truncated_net.modules())[-1].out_channels
        else:
            channel_num = len(channel)
        self.fc = nn.Linear(channel_num * channel_unit_num, fc_out_num)
        self.channel = channel

    def forward(self, x):
        x = self.truncated_net(x)
        if self.channel is not None:
            # extract the specified channel's output
            x = x[:, self.channel]
        x = x.view(x.size(0), -1)  # (batch_num, unit_num)
        x = self.fc(x)
        return x


def dnn_finetuning(netloader,layer,out_class):
    """Fine-tuning the neural network, modifying the 1000 classifier to n classifier
    Parameters:
    -----------
    netloader[Netloader]: class netloader
    layer[str]: fully connected layer name of a DNN network
    out_features[str]:  specify the class of the new dnn model output
    Returns:
    --------
    dnn_model[torchvision.models]: the reconstructed dnn model.
    """
    if 'fc' not in layer:
        raise ValueError('Fine tuning only support to reconstruct fully connected layer')

    dnn_model = netloader.model
    # freeze parameters of pretrained network.
    for param in dnn_model.parameters():
        param.requires_grad = False
    # reconstruct the fully connected layer
    indices = netloader.layer2indices[layer]
    old_classifier = list(netloader.model.children())[-1][:indices[1]]
    in_features = list(netloader.model.children())[-1][indices[1]].in_features
    new_classifier = nn.Sequential(*(list(old_classifier) + [nn.Linear(in_features,out_class)]))

    if hasattr(dnn_model,'classifier'):
        dnn_model.classifier = new_classifier
    elif hasattr(dnn_model,'fc'):
        dnn_model.fc = new_classifier
    return dnn_model