import os
import copy
import h5py
import pytest
import numpy as np

from os.path import join as pjoin
from dnnbrain.io import fileio as fio
from dnnbrain.dnn import core as dcore
from dnnbrain.dnn.base import dnn_mask, array_statistic, dnn_fe

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestStimulus:

    header_true = {
        'type': 'video',
        'path': 'test path',
        'title': 'test title'
    }

    data_true = {
        'stimID': np.array([1, 2, 5, 6]),
        'acc': np.array([0.2, 0.4, 0.1, 0.6]),
        'label': np.array([0, 1, 1, 2])
    }

    def test_init(self):

        # test normal
        stimuli = dcore.Stimulus(self.header_true, self.data_true)
        assert stimuli.header == self.header_true
        for k, v in self.data_true.items():
            np.testing.assert_equal(v, stimuli.get(k))

        # test exception
        data = copy.deepcopy(self.data_true)
        data['acc'] = data['acc'][:-1]
        data['label'] = data['label'][:-1]
        with pytest.raises(AssertionError):
            dcore.Stimulus(data=data)
        data.pop('stimID')
        with pytest.raises(KeyError):
            dcore.Stimulus(data=data)
        with pytest.raises(AssertionError):
            dcore.Stimulus('header')

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
        assert stimuli.header['type'] == type
        assert stimuli.header['title'] == title
        assert stimuli.items == items
        assert stimuli._data['stimID'][[0, 2]].tolist() == stim_ids
        assert stimuli._data['RT'][[0, 2]].tolist() == rts

    def test_save(self):

        # save by Stimulus.save()
        stimuli1 = dcore.Stimulus()
        stimuli1.header = self.header_true
        stimuli1._data = self.data_true
        fname = pjoin(TMP_DIR, 'test.stim.csv')
        stimuli1.save(fname)

        # assert
        stimuli2 = dcore.Stimulus()
        stimuli2.load(fname)
        assert stimuli1.header == stimuli2.header
        for k, v in stimuli1._data.items():
            np.testing.assert_equal(v, stimuli2._data[k])

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

    def test_len(self):

        stimuli = dcore.Stimulus()
        stimuli.header = self.header_true
        stimuli._data = self.data_true
        assert len(stimuli) == len(self.data_true['stimID'])

    def test_getitem(self):

        # prepare
        stimuli = dcore.Stimulus()
        stimuli.header = self.header_true
        stimuli._data = self.data_true

        # -assert int-
        # --assert positive--
        indices = 1
        stim_tmp = stimuli[indices]
        assert stimuli.header == stim_tmp.header
        for k, v in stim_tmp._data.items():
            assert v[0] == stimuli._data[k][indices]

        # --assert negative--
        indices = -1
        stim_tmp = stimuli[indices]
        assert stimuli.header == stim_tmp.header
        for k, v in stim_tmp._data.items():
            assert v[0] == stimuli._data[k][indices]

        # -assert list-
        indices = [0, 2]
        stim_tmp = stimuli[indices]
        assert stimuli.header == stim_tmp.header
        for k, v in stim_tmp._data.items():
            assert np.all(v == stimuli._data[k][indices])

        # -assert slice-
        indices = slice(1, 3)
        stim_tmp = stimuli[indices]
        assert stimuli.header == stim_tmp.header
        for k, v in stim_tmp._data.items():
            assert np.all(v == stimuli._data[k][indices])

        # -assert (list, list of str)-
        indices = ([0, 2], ['stimID', 'label'])
        stim_tmp = stimuli[indices]
        assert stimuli.header == stim_tmp.header
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
            np.testing.assert_equal(data, activation._activation[layer])

    def test_fe(self):

        # prepare
        activation = dcore.Activation()
        activation._activation = self.activation_true
        activation = activation.fe('pca', 3)

        # assert
        for layer, data in self.activation_true.items():
            data = dnn_fe(data, 'pca', 3)
            np.testing.assert_almost_equal(data, activation._activation[layer])

    def test_convolve_hrf(self):

        # prepare
        onsets = np.arange(5)
        durations = np.ones(5)
        n_vol = 2
        tr = 2
        activation = dcore.Activation()
        activation._activation = self.activation_true
        activation = activation.convolve_hrf(onsets, durations, n_vol, tr)

        # assert
        for layer, data in activation._activation.items():
            assert data.shape == (n_vol, *self.activation_true[layer].shape[1:])
            np.testing.assert_almost_equal(data[0], np.zeros(data.shape[1:]))

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


class TestDnnProbe:

    # Prepare DNN activation
    dnn_activ = dcore.Activation()
    dnn_activ.set('conv5', np.random.randn(10, 2, 3, 3))
    dnn_activ.set('fc3', np.random.randn(10, 10, 1, 1))

    def test_probe(self):

        # prepare behavior data
        beh_data = np.random.randint(1, 3, (10, 1))

        # test uv and iter_axis=None
        probe = dcore.DnnProbe(self.dnn_activ, 'uv', 'lrc')
        pred_dict = probe.probe(beh_data)
        assert list(pred_dict.keys()) == self.dnn_activ.layers
        v1_keys = sorted(['score', 'model', 'chn_loc', 'row_loc', 'col_loc'])
        for k1, v1 in pred_dict.items():
            assert sorted(v1.keys()) == v1_keys
            assert np.all(v1['chn_loc'] == 1)
            for v2 in v1.values():
                assert v2.shape == (1, beh_data.shape[1])

        # test mv and iter_axis=channel
        probe.set(model_type='mv', model_name='lrc')
        pred_dict = probe.probe(beh_data, 'channel')
        assert list(pred_dict.keys()) == self.dnn_activ.layers
        v1_keys = sorted(['score', 'model'])
        for k1, v1 in pred_dict.items():
            assert sorted(v1.keys()) == v1_keys
            n_chn = self.dnn_activ.get(k1).shape[1]
            for v2 in v1.values():
                assert v2.shape == (n_chn, beh_data.shape[1])


if __name__ == '__main__':
    pytest.main()
