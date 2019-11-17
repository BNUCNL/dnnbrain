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


class TestSaliencyImage:

    def test_VanlinaSaliencyImage(self):

        # prepare
        dnn = AlexNet()
        transform = Compose([Resize(dnn.img_size), ToTensor()])
        fname = pjoin(DNNBRAIN_TEST, 'image', 'images', 'n02108551_26574.JPEG')
        image = Image.open(fname)
        image.show()
        image = transform(image).unsqueeze(0)

        vanlina = d_algo.VanlinaSaliencyImage(dnn, 'conv1')
        vanlina.set('fc3', 276)
        img_out = vanlina.visualize(image)

        # assert
        assert img_out.shape == image.shape[1:]
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))
        plt.show()
