import os
import h5py
import torch
import pytest
import numpy as np

from PIL import Image
from os.path import join as pjoin
from dnnbrain.dnn import core as dcore
from dnnbrain.dnn import models as db_models
from dnnbrain.dnn.base import VideoClipSet

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestAlexNet:

    def test_save(self):
        pass

    def test_compute_activation(self):

        # -prepare-
        # prepare ground truth
        fname = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.act.h5')
        rf = h5py.File(fname, 'r')

        # prepare stimuli
        stim_file = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.stim.csv')

        # prepare DNN mask
        dmask = dcore.Mask()
        dmask.set('conv5')
        dmask.set('fc3')

        # -compute activation-
        dnn = db_models.AlexNet()
        # compute with Stimulus
        stimuli1 = dcore.Stimulus()
        stimuli1.load(stim_file)
        activation1 = dnn.compute_activation(stimuli1, dmask)

        # compute with ndarray
        stimuli2 = []
        for stim_id in stimuli1.get('stimID'):
            stim_file = pjoin(stimuli1.header['path'], stim_id)
            img = Image.open(stim_file).convert('RGB')
            stimuli2.append(np.asarray(img).transpose((2, 0, 1)))
        stimuli2 = np.asarray(stimuli2)
        activation2 = dnn.compute_activation(stimuli2, dmask)

        # compute with pool
        activation3 = dnn.compute_activation(stimuli1, dmask, 'max')
        activation4 = activation1.pool('max')

        # assert
        np.testing.assert_almost_equal(np.asarray(rf['conv5']),
                                       activation1.get('conv5'), 4)
        np.testing.assert_almost_equal(np.asarray(rf['fc3']),
                                       activation1.get('fc3'), 4)
        np.testing.assert_equal(activation1.get('conv5'), activation2.get('conv5'))
        np.testing.assert_equal(activation1.get('fc3'), activation2.get('fc3'))
        np.testing.assert_equal(activation3.get('conv5'), activation4.get('conv5'))
        np.testing.assert_equal(activation3.get('fc3'), activation4.get('fc3'))

        rf.close()

    def test_get_kernel(self):
        # ground truth
        conv5_shape = torch.tensor((256, 256, 3, 3))
        conv5_1_0 = torch.tensor([[0.0045, -0.0077, -0.0150],
                                  [-0.0303, -0.0441, -0.0176],
                                  [-0.0143, -0.0367, -0.0520]])
        dnn = db_models.AlexNet(False)
        torch.equal(conv5_shape, torch.tensor(dnn.get_kernel('conv5').shape))
        torch.equal(conv5_1_0, dnn.get_kernel('conv5', 1)[0])

    def test_ablate(self):
        pass

    def test_call(self):

        dnn = db_models.AlexNet()
        inputs = torch.randn(2, 3, 224, 224)
        outputs = dnn(inputs)
        assert outputs.shape == (2, 1000)

        
class TestResnet152:

    def test_get_kernel(self):
        conv_shape = torch.tensor((64, 3, 7, 7))
        conv_1_0 = torch.tensor([
            [4.7132e-07, 6.3123e-07, 6.1915e-07, 4.1438e-07,
             2.9313e-07, 2.1123e-07, 1.3036e-07],
            [4.8263e-07, 7.1548e-07, 7.1251e-07, 5.0862e-07,
             3.0581e-07, 2.6611e-07, 2.3413e-07],
            [4.9888e-07, 6.3326e-07, 6.1920e-07, 3.6141e-07,
             1.2629e-07, 1.8429e-07, 2.0732e-07],
            [5.1829e-07, 4.0493e-07, 4.8550e-07, 2.3389e-07,
             1.4707e-07, 2.1979e-07, 2.1960e-07],
            [5.5013e-07, 3.1735e-07, 4.1098e-07, 3.1715e-07,
             3.1079e-07, 3.4928e-07, 3.4718e-07],
            [6.2982e-07, 4.0325e-07, 3.4432e-07, 3.9856e-07,
             4.8297e-07, 6.4529e-07, 5.4214e-07],
            [7.1402e-07, 5.0883e-07, 4.4785e-07, 4.5847e-07,
             6.2946e-07, 6.5617e-07, 5.0979e-07]])
        dnn = db_models.Resnet152(False)
        torch.equal(conv_shape, torch.tensor(dnn.get_kernel('conv').shape))
        torch.equal(conv_1_0, dnn.get_kernel('conv', 1)[0])


class TestVgg19_bn:

    def test_get_kernel(self):
        conv2_shape = torch.tensor((64, 64, 3, 3))
        conv2_1_0 = torch.tensor([
            [6.2157e-09, 1.0084e-08, 1.5116e-08],
            [4.5769e-09, 2.0736e-08, 1.7261e-08],
            [5.1316e-09, 1.1234e-08, 1.7994e-08]])
        dnn = db_models.Vgg19_bn(False)
        torch.equal(conv2_shape, torch.tensor(dnn.get_kernel('conv2').shape))
        torch.equal(conv2_1_0, dnn.get_kernel('conv2', 1)[0])

        conv9_shape = torch.tensor((512, 256, 3, 3))
        conv9_1_0 = torch.tensor([
            [-2.1690e-07, -3.8769e-08, 1.3572e-07],
            [1.1671e-07, -1.8512e-07, -1.5063e-07],
            [9.0605e-08, -2.9362e-08, -1.2601e-07]])
        dnn = db_models.Vgg19_bn(False)
        torch.equal(conv9_shape, torch.tensor(dnn.get_kernel('conv9').shape))
        torch.equal(conv9_1_0, dnn.get_kernel('conv9', 1)[0])

        conv16_shape = torch.tensor((512, 512, 3, 3))
        conv16_1_0 = torch.tensor([
            [-0.0027,  0.0035, -0.0050],
            [-0.0048, -0.0009, -0.0068],
            [-0.0055, -0.0092, -0.0125]])
        dnn = db_models.Vgg19_bn(False)
        torch.equal(conv16_shape, torch.tensor(dnn.get_kernel('conv16').shape))
        torch.equal(conv16_1_0, dnn.get_kernel('conv16', 1)[0])    


class TestR3D:

    def test_compute_activation(self):

        r3d = db_models.R3D()
        layers = ['fc', ('layer1', '0', 'conv1', '0')]
        clip_files = [pjoin(DNNBRAIN_TEST, 'video', 'sub-CSI1_ses-01_imagenet.mp4')]

        # observed
        activ_list1 = r3d.compute_activation(clip_files, layers)

        # ground truth
        activ_list2 = []

        def hook_act(module, input, output):
            acts = output.detach().numpy().copy()
            activ_list2.append(acts)

        handle = r3d.model.layer1[0].conv1[0].register_forward_hook(hook_act)
        vid_clip_set = VideoClipSet(clip_files, r3d.test_transform)
        data, _ = vid_clip_set[0]
        data = torch.unsqueeze(data, 0)
        activ_list2.insert(0, r3d.model(data).detach().numpy())
        handle.remove()

        # test
        for activ1, activ2 in zip(activ_list1, activ_list2):
            assert np.all(activ1 == activ2)


class TestVGGish:

    def test_compute_activation(self):

        vggish = db_models.VGGish(postprocess=False)
        layers = ['fc3', ('features', '11')]
        wavfile = pjoin(DNNBRAIN_TEST, 'bus_chatter.wav')

        # observed
        activ_list1 = vggish.compute_activation(wavfile, layers)

        # ground truth
        activ_list2 = []

        def hook_act(module, input, output):
            acts = output.detach().numpy().copy()
            activ_list2.append(acts)

        handle = vggish.model.features[11].register_forward_hook(hook_act)
        activ_list2.insert(0, vggish(wavfile).detach().numpy())
        handle.remove()

        # test
        for activ1, activ2 in zip(activ_list1, activ_list2):
            assert np.all(activ1 == activ2)

class TestClipResnet:

    def test_compute_activation(self):

        # -prepare-
        # prepare stimuli
        stim_file = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.stim.csv')

        # prepare DNN mask
        dmask = dcore.Mask()
        dmask.set('conv1')
        dmask.set('layer2_block3')
        dmask.set('fc')

        # -compute activation-
        dnn = db_models.ClipResnet()
        # compute with Stimulus
        stimuli1 = dcore.Stimulus()
        stimuli1.load(stim_file)
        activation1 = dnn.compute_activation(stimuli1, dmask)

        # assert
        print(activation1.get('fc').shape)
        # np.testing.assert_equal(activation1.get('conv5'), activation2.get('conv5'))
        # np.testing.assert_equal(activation1.get('fc3'), activation2.get('fc3'))

if __name__ == '__main__':
    pytest.main()
