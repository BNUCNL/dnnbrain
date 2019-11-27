import os

from PIL import Image
from os.path import join as pjoin
from matplotlib import pyplot as plt
from dnnbrain.dnn import algo as d_algo
from dnnbrain.dnn.models import AlexNet
from dnnbrain.utils.util import normalize

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestVanillaSaliencyImage:

    image = Image.open(pjoin(DNNBRAIN_TEST, 'image', 'images', 'n02108551_26574.JPEG'))

    def test_backprop(self):

        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()
        image = dnn.test_transform(self.image).unsqueeze(0)

        # prepare vanilla
        vanilla = d_algo.VanillaSaliencyImage(dnn)
        vanilla.set_layer('fc3', 276)

        # test backprop to conv1
        img_out = vanilla.backprop(image, 'conv1')
        assert img_out.shape == image.shape[1:]
        plt.figure()
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        # test backprop to conv1_relu
        img_out = vanilla.backprop(image, 'conv1_relu')
        assert img_out.shape == (64, 55, 55)
        plt.figure()
        plt.imshow(normalize(img_out)[0])

        plt.show()

    def test_backprop_smooth(self):

        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()
        image = dnn.test_transform(self.image).unsqueeze(0)

        # prepare vanilla
        vanilla = d_algo.VanillaSaliencyImage(dnn)
        vanilla.set_layer('fc3', 276)

        # test backprop to conv1
        img_out = vanilla.backprop_smooth(image, 2, to_layer='conv1')
        assert img_out.shape == image.shape[1:]
        plt.figure()
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        # test backprop to conv1_relu
        img_out = vanilla.backprop_smooth(image, 2, to_layer='conv1_relu')
        assert img_out.shape == (64, 55, 55)
        plt.figure()
        plt.imshow(normalize(img_out)[0])

        plt.show()


class TestGuidedSaliencyImage:

    image = Image.open(pjoin(DNNBRAIN_TEST, 'image', 'images', 'n02108551_26574.JPEG'))

    def test_backprop(self):

        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()
        image = dnn.test_transform(self.image).unsqueeze(0)

        # prepare guided
        guided = d_algo.GuidedSaliencyImage(dnn)

        # -test fc3-
        guided.set_layer('fc3', 276)
        # test backprop to conv1
        img_out = guided.backprop(image, 'conv1')
        assert img_out.shape == image.shape[1:]
        plt.figure()
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        # test backprop to conv1_relu
        img_out = guided.backprop(image, 'conv1_relu')
        assert img_out.shape == (64, 55, 55)
        plt.figure()
        plt.imshow(normalize(img_out)[0])

        # -test conv5-
        guided.set_layer('conv5', 1)
        img_out = guided.backprop(image, 'conv1')
        assert img_out.shape == image.shape[1:]
        plt.figure()
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        plt.show()

    def test_backprop_smooth(self):

        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()
        image = dnn.test_transform(self.image).unsqueeze(0)

        # prepare guided
        guided = d_algo.GuidedSaliencyImage(dnn)
        guided.set_layer('fc3', 276)

        # test backprop to conv1
        img_out = guided.backprop_smooth(image, 2, to_layer='conv1')
        assert img_out.shape == image.shape[1:]
        plt.figure()
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        # test backprop to conv1_relu
        img_out = guided.backprop_smooth(image, 2, to_layer='conv1_relu')
        assert img_out.shape == (64, 55, 55)
        plt.figure()
        plt.imshow(normalize(img_out)[0])

        plt.show()


if __name__ == '__main__':
    tmp = TestGuidedSaliencyImage()
    tmp.test_backprop()
