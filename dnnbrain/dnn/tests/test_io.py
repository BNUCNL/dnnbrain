import os
import sys
import h5py
import torch
import unittest
import pytest
import dnnbrain
import numpy as np

from os.path import join as pjoin
from torchvision import transforms
from dnnbrain.dnn import io

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
HOME = os.path.expanduser('~')
TMP_DIR = pjoin(HOME, '.dnnbrain_tmp')

def test_read_Imagefolder_output(dnnbrain_path):
    """
    Test output of function: read_Imagefolder.
    """
    parpath = os.path.join(dnnbrain_path, 'data', 'test_data', 'picture_folder')
    picpath, conditions = io.read_imagefolder(parpath)
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
    picpath, conditions = io.read_imagefolder(os.path.join(parpath, 'picture_folder'))
    assert len(picpath) == len(conditions)
    # io.generate_stim_csv(parpath, picpath, conditions, outpath=os.path.join(parpath, 'PicStim.csv'))
    
    
def test_PicDataset(dnnbrain_path):
    """
    Test methods and attributes of class PicDataset
    """
    pic_dataset_notransform = io.ImgDataset(os.path.join(dnnbrain_path, 'data', 'test_data', 'PicStim.csv'))
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
    # Test transform method
    pic_dataset_transform = io.ImgDataset(os.path.join(dnnbrain_path, 'data', 'test_data', 'PicStim.csv'),
                                               transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))
    picimg1_new, _ = pic_dataset_transform[0]
    picimg2_new, _ = pic_dataset_transform[2]
    assert picimg1_new.shape == picimg2_new.shape
 
 
def test_NetLoader():
    """
    Test NetLoader
    """
    alexnet_loader = io.NetLoader('alexnet')
    alexnet_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']
    assert alexnet_layers == list(alexnet_loader.layer2indices.keys())
    assert alexnet_loader.img_size == (224,224)
    
    # vgg11_loader = io.NetLoader('vgg11')
    # vgg11_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'fc1', 'fc2', 'fc3']
    # assert vgg11_layers == list(vgg11_loader.layer2indices.keys())
    # assert vgg11_loader.img_size == (224,224)
    # vggface_loader = io.NetLoader('vggface')
    
    # Bad netloader load model, img_size and layer2indices as None
    bad_loader = io.NetLoader('aaa')
    assert bad_loader.model is None
    assert bad_loader.img_size is None
    assert bad_loader.layer2indices is None


class TestActIO(unittest.TestCase):

    def test_ActReader(self):
        # read
        fpath = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.act.h5')
        reader = io.ActReader(fpath)
        rf = h5py.File(fpath, 'r')

        # assert
        self.assertEqual(reader.title, rf.attrs['title'])
        self.assertEqual(reader.cmd, rf.attrs['cmd'])
        self.assertEqual(reader.date, rf.attrs['date'])
        self.assertEqual(reader.layers, list(rf.keys()))
        self.assertTrue(np.all(reader.get_attr('conv5', 'raw_shape') ==
                               rf['conv5'].attrs['raw_shape']))
        self.assertTrue(np.all(reader.get_act('fc3') == np.array(rf['fc3'])))

        reader.close()
        rf.close()

    def test_ActWriter(self):
        # prepare variates
        fpath = pjoin(TMP_DIR, 'test.act.h5')
        title = 'test title'
        layers = ['test layer']
        act = np.random.randn(2, 3)
        raw_shape = act.shape

        # write
        writer = io.ActWriter(fpath, title)
        writer.set_act(layers[0], act)
        writer.set_attr(layers[0], 'raw_shape', raw_shape)
        writer.close()

        # compare
        rf = h5py.File(fpath, 'r')
        self.assertEqual(title, rf.attrs['title'])
        self.assertEqual(' '.join(sys.argv), rf.attrs['cmd'])
        self.assertEqual(layers, list(rf.keys()))
        self.assertTrue(np.all(np.array(raw_shape) ==
                               rf[layers[0]].attrs['raw_shape']))
        self.assertTrue(np.all(act == np.array(rf[layers[0]])))
        rf.close()


if __name__ == '__main__':
    unittest.main()
