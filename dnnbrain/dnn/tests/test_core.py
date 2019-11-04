import os
import pytest

from os.path import join as pjoin
from dnnbrain.io import fileio as fio
from dnnbrain.dnn import core as dcore

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestStimulus:

    def test_load(self):
        pass

    def test_save(self):
        pass

    def test_get(self):
        pass

    def test_set(self):
        pass

    def test_delete(self):
        pass

    def test_getitem(self):
        pass


class TestDNN:

    def test_load(self):
        pass

    def test_save(self):
        pass

    def test_compute_activation(self):
        pass

    def test_get_kernel(self):
        pass

    def test_ablate(self):
        pass


class TestActivation:

    def test_load(self):
        pass

    def test_save(self):
        pass

    def test_get(self):
        pass

    def test_set(self):
        pass

    def test_delete(self):
        pass

    def test_concatenate(self):
        pass

    def test_mask(self):
        pass

    def test_pool(self):
        pass

    def test_fe(self):
        pass

    def test_arithmetic(self):
        pass


class TestMask:

    def test_get(self):

        fname = pjoin(DNNBRAIN_TEST, 'alexnet.dmask.csv')

        # ground truth
        dmask_dict = fio.MaskFile(fname).read()

        # load by Mask.load()
        dmask = dcore.Mask(fname)

        # assert
        assert dmask.layers == list(dmask_dict.keys())
        for layer in dmask.layers:
            assert dmask.get(layer) == dmask_dict[layer]

    def test_set(self):

        # ground truth
        dmask_dict = {
            'conv1': {'col': [1, 2, 3]},
            'conv2': {'row': [2, 5, 6]},
            'fc1': {'chn': [2, 4, 6]}
        }

        # set by Mask.set()
        dmask = dcore.Mask()
        for layer, d in dmask_dict.items():
            dmask.set(layer, channels=d.get('chn'), rows=d.get('row'), columns=d.get('col'))

        # assert
        assert dmask.layers == list(dmask_dict.keys())
        for layer in dmask.layers:
            assert dmask.get(layer) == dmask_dict[layer]

    def test_copy(self):

        # prepare origin dmask
        dmask_dict = {
            'conv1': {'col': [1, 2, 3]},
            'conv2': {'row': [2, 5, 6]},
            'fc1': {'chn': [2, 4, 6]}
        }
        dmask = dcore.Mask()
        for layer, d in dmask_dict.items():
            dmask.set(layer, channels=d.get('chn'), rows=d.get('row'), columns=d.get('col'))

        # make a copy
        dmask_copy = dmask.copy()

        # assert
        assert dmask.layers == dmask_copy.layers
        for layer in dmask.layers:
            assert dmask.get(layer) == dmask_copy.get(layer)

    def test_delete(self):

        # prepare a dmask
        dmask_dict = {
            'conv1': {'col': [1, 2, 3]},
            'conv2': {'row': [2, 5, 6]},
            'fc1': {'chn': [2, 4, 6]}
        }
        dmask = dcore.Mask()
        for layer, d in dmask_dict.items():
            dmask.set(layer, channels=d.get('chn'), rows=d.get('row'), columns=d.get('col'))

        # delete a layer
        layer_del = 'conv1'
        dmask.delete(layer_del)

        # assert
        dmask_dict.pop(layer_del)
        assert dmask.layers == list(dmask_dict.keys())
        for layer in dmask.layers:
            assert dmask.get(layer) == dmask_dict[layer]


if __name__ == '__main__':
    pytest.main()
