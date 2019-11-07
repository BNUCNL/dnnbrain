import os
import h5py
import pytest
import numpy as np

from os.path import join as pjoin
from dnnbrain.io import fileio as fio

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestStimulusFile:

    def test_read(self):

        # ground truth
        stim_type = 'image'
        stim_title = 'ImageNet images in all 5000scenes runs of sub-CSI1_ses-01'
        stim_id = ['n01930112_19568.JPEG', 'n03733281_29214.JPEG']
        stim_rt = [3.6309, 4.2031]

        # load by StimulusFile.read()
        stim_file = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.stim.csv')
        stim = fio.StimulusFile(stim_file).read()

        # compare
        assert stim['type'] == stim_type
        assert stim['title'] == stim_title
        assert stim['data']['stimID'][:2].tolist() == stim_id
        assert stim['data']['RT'][:2].tolist() == stim_rt

    def test_write(self):

        # ground truth
        type = 'video'
        path = 'video_file'
        title = 'video stimuli'
        data = {'stimID': [1, 3, 5]}

        # save by StimulusFile.write()
        fname = pjoin(TMP_DIR, 'test.stim.csv')
        fio.StimulusFile(fname).write(type, path, data, title=title)

        # compare
        stim_dict = fio.StimulusFile(fname).read()
        assert stim_dict['type'] == type
        assert stim_dict['path'] == path
        assert stim_dict['title'] == title
        assert stim_dict['data']['stimID'].tolist() == data['stimID']


class TestActivationFile:

    def test_read(self):

        fname = pjoin(DNNBRAIN_TEST, "image", "sub-CSI1_ses-01_imagenet.act.h5")
        # ground truth
        rf = h5py.File(fname, 'r')

        # load by ActivationFile.read()
        activation = fio.ActivationFile(fname).read()

        # assert
        assert np.all(activation['conv5'] == np.array(rf['conv5']))
        assert np.all(activation['fc3'] == np.array(rf['fc3']))

        rf.close()

    def test_write(self):

        fname = pjoin(TMP_DIR, 'test.act.h5')
        # ground truth
        activation = {
            'conv5': np.random.randn(5, 3, 13, 13),
            'fc3': np.random.randn(5, 10, 1, 1)
        }

        # save by ActivationFile.write()
        fio.ActivationFile(fname).write(activation)

        # assert
        rf = h5py.File(fname, 'r')
        assert list(activation.keys()) == list(rf.keys())
        for layer, data in activation.items():
            assert np.all(data == np.array(rf[layer]))

        rf.close()


class TestMaskFile:

    def test_read(self):

        # ground truth
        conv5_chn = [1, 2, 3]
        conv5_row = [4, 5]
        conv5_col = [6, 7, 8]
        fc3_chn = [1, 2, 3]
        fc3_keys = ['chn']

        # load by MaskFile.read()
        fname = pjoin(DNNBRAIN_TEST, 'alexnet.dmask.csv')
        dmask = fio.MaskFile(fname).read()

        # assert
        assert dmask['conv5']['chn'] == conv5_chn
        assert dmask['conv5']['row'] == conv5_row
        assert dmask['conv5']['col'] == conv5_col
        assert dmask['fc3']['chn'] == fc3_chn
        assert list(dmask['fc3'].keys()) == fc3_keys

    def test_write(self):

        # ground truth
        dmask1 = {
            'conv1': {'col': [1, 2, 3]},
            'conv2': {'row': [2, 5, 6]},
            'fc1': {'chn': [2, 4, 6]}
        }

        # save by MaskFile.write()
        fname = pjoin(TMP_DIR, 'test.dmask.csv')
        fio.MaskFile(fname).write(dmask1)

        # assert
        dmask2 = fio.MaskFile(fname).read()
        assert dmask1.keys() == dmask2.keys()
        assert dmask1['conv1']['col'] == dmask2['conv1']['col']
        assert dmask1['conv2']['row'] == dmask2['conv2']['row']
        assert dmask1['fc1']['chn'] == dmask2['fc1']['chn']
        assert dmask1['fc1'].keys() == dmask2['fc1'].keys()


if __name__ == '__main__':
    pytest.main()
