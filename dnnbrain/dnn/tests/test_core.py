import os
import copy
import h5py
import pytest
import numpy as np

from PIL import Image
from os.path import join as pjoin
from dnnbrain.io import fileio as fio
from dnnbrain.dnn import core as dcore
from dnnbrain.dnn.base import dnn_mask, array_statistic, dnn_fe

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestStimulus:

    meta_true = {
        'type': 'video',
        'path': 'test path',
        'title': 'test title'
    }

    data_true = {
        'stimID': np.array([1, 2, 5, 6]),
        'acc': np.array([0.2, 0.4, 0.1, 0.6]),
        'label': np.array([0, 1, 1, 2])
    }

    def test_load(self):

        # ground truth
        type = 'image'
        title = 'ImageNet images in all 5000scenes runs of sub-CSI1_ses-01'
        items = ['stimID', 'label', 'onset', 'duration', 'response', 'RT']
        stim_ids = ['n01930112_19568.JPEG', 'n07695742_5848.JPEG']
        rts = [3.6309, 1.8948]

        # load by Stimulus.load()
        fname = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.stim.csv')
        stimuli = dcore.Stimulus()
        stimuli.load(fname)

        # assert
        assert stimuli.meta['type'] == type
        assert stimuli.meta['title'] == title
        assert stimuli.items == items
        assert stimuli._data['stimID'][[0, 2]].tolist() == stim_ids
        assert stimuli._data['RT'][[0, 2]].tolist() == rts

    def test_save(self):

        # save by Stimulus.save()
        stimuli1 = dcore.Stimulus()
        stimuli1.meta = self.meta_true
        stimuli1._data = self.data_true
        fname = pjoin(TMP_DIR, 'test.stim.csv')
        stimuli1.save(fname)

        # assert
        stimuli2 = dcore.Stimulus(fname)
        assert stimuli1.meta == stimuli2.meta
        for k, v in stimuli1._data.items():
            assert v.tolist() == stimuli2._data[k].tolist()

    def test_get(self):

        # prepare
        stimuli = dcore.Stimulus()
        stimuli._data = self.data_true

        # assert
        for k, v in stimuli._data.items():
            assert np.all(v == stimuli.get(k))

    def test_set(self):

        # set by Stimulus.set()
        stimuli = dcore.Stimulus()
        for k, v in self.data_true.items():
            stimuli.set(k, v)

        # assert
        for k, v in self.data_true.items():
            assert np.all(v == stimuli._data[k])

    def test_delete(self):

        # prepare
        data = copy.deepcopy(self.data_true)
        stimuli = dcore.Stimulus()
        stimuli._data = copy.deepcopy(self.data_true)

        # delete by Stimulus.delete()
        stimuli.delete('acc')

        # assert
        data.pop('acc')
        for k, v in data.items():
            assert np.all(v == stimuli._data[k])

    def test_getitem(self):

        # prepare
        stimuli = dcore.Stimulus()
        stimuli.meta = self.meta_true
        stimuli._data = self.data_true

        # -assert int-
        # --assert positive--
        indices = 1
        stim_tmp = stimuli[indices]
        assert stimuli.meta == stim_tmp.meta
        for k, v in stim_tmp._data.items():
            assert v[0] == stimuli._data[k][indices]

        # --assert negative--
        indices = -1
        stim_tmp = stimuli[indices]
        assert stimuli.meta == stim_tmp.meta
        for k, v in stim_tmp._data.items():
            assert v[0] == stimuli._data[k][indices]

        # -assert list-
        indices = [0, 2]
        stim_tmp = stimuli[indices]
        assert stimuli.meta == stim_tmp.meta
        for k, v in stim_tmp._data.items():
            assert np.all(v == stimuli._data[k][indices])

        # -assert slice-
        indices = slice(1, 3)
        stim_tmp = stimuli[indices]
        assert stimuli.meta == stim_tmp.meta
        for k, v in stim_tmp._data.items():
            assert np.all(v == stimuli._data[k][indices])

        # -assert (list, list of str)-
        indices = ([0, 2], ['stimID', 'label'])
        stim_tmp = stimuli[indices]
        assert stimuli.meta == stim_tmp.meta
        for k in indices[1]:
            assert np.all(stim_tmp._data[k] == stimuli._data[k][indices[0]])

        # -assert ()-
        # -assert (int,)-
        # -assert (list,)-
        # -assert (slice,)-
        # -assert (int, int)-
        # -assert (int, str)-
        # -assert (int, list of int)-
        # -assert (int, list of str)-
        # -assert (int, slice)-
        # -assert (list, int)-
        # -assert (list, str)-
        # -assert (list, list of int)-
        # -assert (list, slice)-
        # -assert (slice, int)-
        # -assert (slice, str)-
        # -assert (slice, list of int)-
        # -assert (slice, list of str)-
        # -assert (slice, slice)-


class TestDNN:

    def test_load(self):
        pass

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
        dnn = dcore.DNN('alexnet')
        # compute with Stimulus
        stimuli1 = dcore.Stimulus(stim_file)
        activation1 = dnn.compute_activation(stimuli1, dmask)

        # compute with ndarray
        stimuli2 = []
        for stim_id in stimuli1.get('stimID'):
            stim_file = pjoin(stimuli1.meta['path'], stim_id)
            img = Image.open(stim_file).convert('RGB')
            stimuli2.append(np.asarray(img).transpose((2, 0, 1)))
        stimuli2 = np.asarray(stimuli2)
        activation2 = dnn.compute_activation(stimuli2, dmask)

        # assert
        np.testing.assert_almost_equal(np.asarray(rf['conv5']),
                                       activation1.get('conv5'), 4)
        np.testing.assert_almost_equal(np.asarray(rf['fc3']),
                                       activation1.get('fc3'), 4)
        np.testing.assert_equal(activation1.get('conv5'), activation2.get('conv5'))
        np.testing.assert_equal(activation1.get('fc3'), activation2.get('fc3'))

        rf.close()

    def test_get_kernel(self):
        pass

    def test_ablate(self):
        pass


class TestActivation:

    activation_true = {
        'conv5': np.random.randn(5, 3, 13, 13),
        'fc3': np.random.randn(5, 10, 1, 1)
    }

    def test_load(self):

        # ground truth
        fname = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.act.h5')
        rf = h5py.File(fname, 'r')

        # prepare dmask
        dmask = dcore.Mask()
        dmask.set('conv5', channels=[1, 2, 3], rows=[4, 5],
                  columns=[6, 7, 8])
        dmask.set('fc3', channels=[1, 2, 3])

        # assert without dmask
        activation = dcore.Activation(fname)
        for layer in rf.keys():
            np.testing.assert_array_equal(np.asarray(rf[layer]),
                                          activation._activation[layer])

        # assert with dmask
        activation.load(fname, dmask)
        for layer in rf.keys():
            mask = dmask.get(layer)
            data_true = dnn_mask(np.asarray(rf[layer]), mask.get('chn'),
                                 mask.get('row'), mask.get('col'))
            np.testing.assert_array_equal(data_true,
                                          activation._activation[layer])
        rf.close()

    def test_save(self):

        # save by Activation.save()
        fname = pjoin(TMP_DIR, 'test.act.h5')
        activation = dcore.Activation()
        activation._activation = self.activation_true
        activation.save(fname)

        # assert
        rf = h5py.File(fname, 'r')
        for layer, data in self.activation_true.items():
            np.testing.assert_array_equal(data,
                                          np.asarray(rf[layer]))
        rf.close()

    def test_get(self):
        pass

    def test_set(self):
        pass

    def test_delete(self):
        pass

    def test_concatenate(self):

        # prepare
        activation1 = dcore.Activation()
        activation1._activation = self.activation_true
        activation2 = dcore.Activation()
        activation2._activation = self.activation_true
        activation = activation1.concatenate([activation2])

        # assert
        for layer, data in self.activation_true.items():
            data = np.concatenate([data, data])
            np.testing.assert_array_equal(data,
                                          activation._activation[layer])

    def test_mask(self):

        # prepare
        dmask = dcore.Mask()
        dmask.set('conv5', channels=[1, 2], rows=[3, 4],
                  columns=[5, 6])
        dmask.set('fc3', channels=[1, 2, 3])
        activation = dcore.Activation()
        activation._activation = self.activation_true
        activation = activation.mask(dmask)

        # assert
        for layer, data in self.activation_true.items():
            mask = dmask.get(layer)
            data = dnn_mask(data, mask.get('chn'),
                                 mask.get('row'), mask.get('col'))
            np.testing.assert_array_equal(data,
                                          activation._activation[layer])

    def test_pool(self):

        # prepare
        activation = dcore.Activation()
        activation._activation = self.activation_true
        activation = activation.pool('max')

        # assert
        for layer, data in self.activation_true.items():
            data = array_statistic(data, 'max', (2, 3), True)
            np.testing.assert_array_equal(data,
                                          activation._activation[layer])

    def test_fe(self):

        # prepare
        activation = dcore.Activation()
        activation._activation = self.activation_true
        activation = activation.fe('pca', 3)

        # assert
        for layer, data in self.activation_true.items():
            data = dnn_fe(data, 'pca', 3)
            np.testing.assert_almost_equal(data,
                                           activation._activation[layer])

    def test_arithmetic(self):
        pass

    def test_getitem(self):

        # prepare
        act = dcore.Activation()
        act._activation = self.activation_true

        # -assert int-
        # assert positive
        indices = 1
        act_tmp = act[indices]
        for layer, data in act_tmp._activation.items():
            np.testing.assert_equal(data[0], act.get(layer)[indices])

        # assert negative
        indices = -1
        act_tmp = act[indices]
        for layer, data in act_tmp._activation.items():
            np.testing.assert_equal(data[0], act.get(layer)[indices])

        # -assert list-
        indices = [1, 3]
        act_tmp = act[indices]
        for layer, data in act_tmp._activation.items():
            np.testing.assert_equal(data, act.get(layer)[indices])

        # -assert slice-
        indices = slice(1, 3)
        act_tmp = act[indices]
        for layer, data in act_tmp._activation.items():
            np.testing.assert_equal(data, act.get(layer)[indices])


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


class TestEncoder:

    # Prepare some simulation data
    activation = dcore.Activation()
    activation.set('conv5', np.random.randn(10, 2, 3, 3))
    activation.set('fc3', np.random.randn(10, 10, 1, 1))
    response = np.random.randn(10, 2)

    def test_uva(self):

        v1_keys = sorted(['score', 'channel', 'row', 'column', 'model'])
        # assert when iter_axis is None
        encoder = dcore.Encoder('glm')
        pred_dict = encoder.uva(self.activation, self.response)
        assert list(pred_dict.keys()) == self.activation.layers
        for v1 in pred_dict.values():
            assert sorted(v1.keys()) == v1_keys
            for v2 in v1.values():
                assert v2.shape == (1, self.response.shape[1])

        # assert when iter_axis is channel
        encoder.set(iter_axis='channel')
        pred_dict = encoder.uva(self.activation, self.response)
        assert list(pred_dict.keys()) == self.activation.layers
        for k1, v1 in pred_dict.items():
            assert sorted(v1.keys()) == v1_keys
            n_chn = self.activation.get(k1).shape[1]
            for v2 in v1.values():
                assert v2.shape == (n_chn, self.response.shape[1])

        # assert when iter_axis is row_col
        encoder.set(iter_axis='row_col')
        pred_dict = encoder.uva(self.activation, self.response)
        assert list(pred_dict.keys()) == self.activation.layers
        for k1, v1 in pred_dict.items():
            assert sorted(v1.keys()) == v1_keys
            n_row, n_col = self.activation.get(k1).shape[2:]
            n_row_col = n_row * n_col
            for v2 in v1.values():
                assert v2.shape == (n_row_col, self.response.shape[1])

    def test_mva(self):

        v1_keys = sorted(['score', 'model'])
        # assert when iter_axis is None
        encoder = dcore.Encoder('glm')
        pred_dict = encoder.mva(self.activation, self.response)
        assert list(pred_dict.keys()) == self.activation.layers
        for v1 in pred_dict.values():
            assert sorted(v1.keys()) == v1_keys
            for v2 in v1.values():
                assert v2.shape == (1, self.response.shape[1])

        # assert when iter_axis is channel
        encoder.set(iter_axis='channel')
        pred_dict = encoder.mva(self.activation, self.response)
        assert list(pred_dict.keys()) == self.activation.layers
        for k1, v1 in pred_dict.items():
            assert sorted(v1.keys()) == v1_keys
            n_chn = self.activation.get(k1).shape[1]
            for v2 in v1.values():
                assert v2.shape == (n_chn, self.response.shape[1])

        # assert when iter_axis is row_col
        encoder.set(iter_axis='row_col')
        pred_dict = encoder.mva(self.activation, self.response)
        assert list(pred_dict.keys()) == self.activation.layers
        for k1, v1 in pred_dict.items():
            assert sorted(v1.keys()) == v1_keys
            n_row, n_col = self.activation.get(k1).shape[2:]
            n_row_col = n_row * n_col
            for v2 in v1.values():
                assert v2.shape == (n_row_col, self.response.shape[1])


class TestDecoder:

    # Prepare some simulation data
    activation = dcore.Activation()
    activation.set('conv5', np.random.randn(10, 2, 3, 3))
    activation.set('fc3', np.random.randn(10, 10, 1, 1))
    response = np.random.randn(10, 2)

    def test_uva(self):

        v1_keys = sorted(['score', 'measurement', 'model'])
        decoder = dcore.Decoder('glm')
        pred_dict = decoder.uva(self.response, self.activation)
        assert list(pred_dict.keys()) == self.activation.layers
        for k1, v1 in pred_dict.items():
            assert sorted(v1.keys()) == v1_keys
            _, n_chn, n_row, n_col = self.activation.get(k1).shape
            for v2 in v1.values():
                assert v2.shape == (n_chn, n_row, n_col)

    def test_mva(self):

        v1_keys = sorted(['score', 'model'])
        decoder = dcore.Decoder('glm')
        pred_dict = decoder.mva(self.response, self.activation)
        assert list(pred_dict.keys()) == self.activation.layers
        for k1, v1 in pred_dict.items():
            assert sorted(v1.keys()) == v1_keys
            _, n_chn, n_row, n_col = self.activation.get(k1).shape
            for v2 in v1.values():
                assert v2.shape == (n_chn, n_row, n_col)


if __name__ == '__main__':
    pytest.main()
