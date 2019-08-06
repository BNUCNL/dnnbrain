import pytest
import dnnbrain
from dnnbrain.dnn import io as iofiles
import os

import torch
from torchvision import transforms


def test_read_Imagefolder_output(dnnbrain_path):
    """
    Test output of function: read_Imagefolder.
    """
    parpath = os.path.join(dnnbrain_path, 'data', 'test_data', 'picture_folder')
    picpath, conditions = iofiles.read_Imagefolder(parpath)
    # output type
    assert isinstance(picpath, list)
    assert isinstance(conditions, list)
    # output correction
    picname_actual = ['ant1.jpg', 'ant2.jpg', 'bees1.jpg', 'bees2.jpg']
    conditions_actual = ['ants', 'ants', 'bees', 'bees']
    picpath_actual = [os.path.join(conditions[i], picname_actual[i]) for i, _ in enumerate(picname_actual)]
    assert conditions == conditions_actual
    assert picpath == picpath_actual
    

@pytest.mark.skipif(os.path.isfile(os.path.join(os.path.dirname(dnnbrain.__file__), 'data', 'test_data', 'PicStim.csv')), 
                    reason='PicSim.csv has been stored in dnnbrain')
def test_generate_stim_csv(dnnbrain_path):
    """
    Test output of generate_stim_csv
    """
    parpath = os.path.join(dnnbrain_path, 'data', 'test_data')
    # Calling function to generate_stim_csv
    picpath, conditions = iofiles.read_Imagefolder(os.path.join(parpath, 'picture_folder'))
    assert len(picpath) == len(conditions)
    iofiles.generate_stim_csv(parpath, picpath, conditions, outpath=os.path.join(parpath, 'PicStim.csv'))
    
    
def test_PicDataset(dnnbrain_path):
    """
    Test methods and attributes of class PicDataset
    """
    pic_dataset_notransform = iofiles.PicDataset(os.path.join(dnnbrain_path, 'data', 'test_data', 'PicStim.csv'))
    # Test method of __len__
    assert len(pic_dataset_notransform) == 4
    picimg1, target_label1 = pic_dataset_notransform[0]
    picimg2, target_label2 = pic_dataset_notransform[2]
    # Test shape of picimg
    assert picimg1.shape != picimg2.shape
    # Test target label
    assert target_label1 == 0
    assert target_label2 == 1
    # Test output type
    assert isinstance(picimg1, torch.Tensor)
    assert isinstance(picimg2, torch.Tensor)
    # Test method get_picname
    picname, condition = pic_dataset_notransform.get_picname(0)
    assert picname == 'ant1.jpg'
    assert condition == 'ants'    
    # Test transform method
    pic_dataset_transform = iofiles.PicDataset(os.path.join(dnnbrain_path, 'data', 'test_data', 'PicStim.csv'), 
                                               transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))
    picimg1_new, _ = pic_dataset_transform[0]
    picimg2_new, _ = pic_dataset_transform[2]
    assert picimg1_new.shape == picimg2_new.shape
 
 
def test_NetLoader():
    """
    Test NetLoader
    """
    alexnet_loader = iofiles.NetLoader('alexnet')
    alexnet_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']
    assert alexnet_layers == list(alexnet_loader.layer2indices.keys())
    assert alexnet_loader.img_size == (224,224)
    
    # vgg11_loader = iofiles.NetLoader('vgg11')
    # vgg11_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'fc1', 'fc2', 'fc3']
    # assert vgg11_layers == list(vgg11_loader.layer2indices.keys())
    # assert vgg11_loader.img_size == (224,224)
    # vggface_loader = iofiles.NetLoader('vggface')
    
    # Bad netloader load model, img_size and layer2indices as None
    bad_loader = iofiles.NetLoader('aaa')
    assert bad_loader.model is None
    assert bad_loader.img_size is None
    assert bad_loader.layer2indices is None  