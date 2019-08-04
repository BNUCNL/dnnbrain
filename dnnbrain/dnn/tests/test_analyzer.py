# Test module of analyzer
import pytest
import os
from dnnbrain.dnn import analyzer
from dnnbrain.dnn import io as iofiles
from torch.utils.data import dataloader
from torchvision import transforms

def test_dnn_activation(dnnbrain_path):
    """
    Test dnn_activation
    """
    # Prepare data as a dataloader
    netname = 'alexnet'
    alex_netloader = iofiles.NetLoader(netname)
    transform = transforms.Compose([transforms.Resize(alex_netloader.img_size), transforms.ToTensor()])
    pic_dataset = iofiles.PicDataset(os.path.join(dnnbrain_path, 'data', 'test_data', 'PicStim.csv'), transform = transform)
    pic_dataloader = dataloader.DataLoader(pic_dataset, batch_size=2, shuffle=False)
    dnnact_allchannel_layer1 = analyzer.dnn_activation(pic_dataloader, netname, 'conv1')
    assert dnnact_allchannel_layer1.shape == (4,64,55,55)
    dnnact_channel1_layer1 = analyzer.dnn_activation(pic_dataloader, netname, 'conv1', channel=[1])
    assert dnnact_channel1_layer1.shape == (4,1,55,55)