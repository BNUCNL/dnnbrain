import os
import sys
import cv2
import h5py
import unittest
import pytest
import dnnbrain
import numpy as np

from os.path import join as pjoin
from PIL import Image
from torchvision import transforms
from dnnbrain.dnn import io as dio

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')


def test_read_Imagefolder_output(dnnbrain_path):
    """
    Test output of function: read_Imagefolder.
    """
    parpath = os.path.join(dnnbrain_path, 'data', 'test_data', 'picture_folder')
    picpath, conditions = dio.read_imagefolder(parpath)
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
    picpath, conditions = dio.read_imagefolder(os.path.join(parpath, 'picture_folder'))
    assert len(picpath) == len(conditions)
    # io.generate_stim_csv(parpath, picpath, conditions, outpath=os.path.join(parpath, 'PicStim.csv'))

 
def test_NetLoader():
    """
    Test NetLoader
    """
    alexnet_loader = dio.NetLoader('alexnet')
    alexnet_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']
    assert alexnet_layers == list(alexnet_loader.layer2indices.keys())
    assert alexnet_loader.img_size == (224,224)
    
    # vgg11_loader = io.NetLoader('vgg11')
    # vgg11_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'fc1', 'fc2', 'fc3']
    # assert vgg11_layers == list(vgg11_loader.layer2indices.keys())
    # assert vgg11_loader.img_size == (224,224)
    # vggface_loader = io.NetLoader('vggface')
    
    # Bad netloader load model, img_size and layer2indices as None
    bad_loader = dio.NetLoader('aaa')
    assert bad_loader.model is None
    assert bad_loader.img_size is None
    assert bad_loader.layer2indices is None


class TestActIO(unittest.TestCase):

    def test_ActReader(self):
        # read
        fpath = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.act.h5')
        reader = dio.ActReader(fpath)
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
        writer = dio.ActWriter(fpath, title)
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


class TestDataset(unittest.TestCase):

    def test_ImgDataset(self):

        # prepare variates
        img_par = pjoin(DNNBRAIN_TEST, 'image', 'images')
        img_ids = ['n01443537_2819.JPEG', 'n01531178_2651.JPEG']
        labels = [3, 2]
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])

        # assert
        dataset = dio.ImgDataset(img_par, img_ids, labels, transform)

        img0, label0 = dataset[0]
        img_0 = transform(Image.open(pjoin(img_par, img_ids[0])))
        self.assertTrue(np.all(img0.numpy() == img_0.numpy()))
        self.assertEqual(label0, labels[0])

        img1, label1 = dataset[1]
        img_1 = transform(Image.open(pjoin(img_par, img_ids[1])))
        self.assertTrue(np.all(img1.numpy() == img_1.numpy()))
        self.assertEqual(label1, labels[1])

    def test_VidDataset(self):

        # prepare variates
        vid_file = pjoin(DNNBRAIN_TEST, 'video', 'sub-CSI1_ses-01_imagenet.mp4')
        frame_nums = [1, 2]
        labels = [3, 2]
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])

        # assert
        dataset = dio.VidDataset(vid_file, frame_nums, labels, transform)
        vid_cap = cv2.VideoCapture(vid_file)

        frame0, label0 = dataset[0]
        _, frame_0 = vid_cap.read()
        frame_0 = Image.fromarray(cv2.cvtColor(frame_0, cv2.COLOR_BGR2RGB))
        frame_0 = transform(frame_0)
        self.assertTrue(np.all(frame0.numpy() == frame_0.numpy()))
        self.assertEqual(label0, labels[0])

        frame1, label1 = dataset[1]
        _, frame_1 = vid_cap.read()
        frame_1 = Image.fromarray(cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB))
        frame_1 = transform(frame_1)
        self.assertTrue(np.all(frame1.numpy() == frame_1.numpy()))
        self.assertEqual(label1, labels[1])


if __name__ == '__main__':
    unittest.main()
