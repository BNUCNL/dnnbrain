from torch import nn


class DNN2BrainNet(nn.Module):

    def __init__(self, truncated_net, channel_unit_num, fc_out_num, channel=None):
        """

        :param truncated_net: torch.nn.Module
            a truncated neural network from the pretrained network
        :param channel_unit_num: int
            the number of units of each channel of the last layer in the truncated network
        :param fc_out_num: int
            the number of the out channels of the full connection layer
        :param channel: int
            The sequence number of the out channel of the selected layer
        """
        super(DNN2BrainNet, self).__init__()
        self.truncated_net = truncated_net
        if channel is None:
            channel_num = list(self.truncated_net.modules())[-1].out_channels
        else:
            channel_num = 1
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
