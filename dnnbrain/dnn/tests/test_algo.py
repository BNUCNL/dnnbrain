import os

from PIL import Image
from os.path import join as pjoin
from matplotlib import pyplot as plt
from dnnbrain.dnn import algo as d_algo
from dnnbrain.dnn.models import AlexNet
from dnnbrain.utils.util import normalize
from torchvision.transforms import Compose, Resize, ToTensor

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestVanillaSaliencyImage:

    def test_backprop(self):

        # prepare DNN
        dnn = AlexNet()

        # prepare image
        transform = Compose([Resize(dnn.img_size), ToTensor()])
        fname = pjoin(DNNBRAIN_TEST, 'image', 'images', 'n02108551_26574.JPEG')
        image = Image.open(fname)
        image.show()
        image = transform(image).unsqueeze(0)

        # prepare vanilla
        vanilla = d_algo.VanillaSaliencyImage(dnn)
        vanilla.set_layer('fc3', 276)

        # test backprop to conv1
        img_out = vanilla.backprop(image, 'conv1')
        assert img_out.shape == image.shape[1:]
        plt.figure('conv1')
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        # test backprop to conv1_relu
        img_out = vanilla.backprop(image, 'conv1_relu')
        assert img_out.shape == (64, 55, 55)
        plt.figure('conv1_relu')
        plt.imshow(normalize(img_out)[0])

        plt.show()

    def test_backprop_smooth(self):

        # prepare DNN
        dnn = AlexNet()

        # prepare image
        transform = Compose([Resize(dnn.img_size), ToTensor()])
        fname = pjoin(DNNBRAIN_TEST, 'image', 'images', 'n02108551_26574.JPEG')
        image = Image.open(fname)
        image.show()
        image = transform(image).unsqueeze(0)

        # prepare vanilla
        vanilla = d_algo.VanillaSaliencyImage(dnn)
        vanilla.set_layer('fc3', 276)

        # test backprop to conv1
        img_out = vanilla.backprop_smooth(image, 30, to_layer='conv1')
        assert img_out.shape == image.shape[1:]
        plt.figure('conv1')
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        # test backprop to conv1_relu
        img_out = vanilla.backprop_smooth(image, 30, to_layer='conv1_relu')
        assert img_out.shape == (64, 55, 55)
        plt.figure('conv1_relu')
        plt.imshow(normalize(img_out)[0])

        plt.show()
