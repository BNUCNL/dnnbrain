import os
import h5py
import pytest
import numpy as np

from os.path import join as pjoin
from dnnbrain.brain.core import ROI, BrainEncoder, BrainDecoder
from dnnbrain.dnn.core import Activation

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestROI:

    rois_true = ['OFA', 'FFA', 'PPA']
    data_true = np.random.randn(5, 3)

    def test_load(self):

        fname = pjoin(DNNBRAIN_TEST, 'PHA1.roi.h5')
        # ground truth
        rf = h5py.File(fname, 'r')

        # assert read without rois
        roi = ROI()
        roi.load(fname)
        assert roi.rois == rf.attrs['roi'].tolist()
        np.testing.assert_equal(roi.data, rf['data'][:])

        # assert read with rois
        roi = ROI()
        roi.load(fname, 'PHA1_R')
        assert roi.rois == ['PHA1_R']
        np.testing.assert_equal(roi.data, rf['data'][:, [1]])

        rf.close()

    def test_save(self):

        # save by ROI::save
        fname = pjoin(TMP_DIR, 'test.roi.h5')
        roi = ROI()
        roi.rois = self.rois_true
        roi.data = self.data_true
        roi.save(fname)

        # assert
        rf = h5py.File(fname, 'r')
        assert self.rois_true == rf.attrs['roi'].tolist()
        np.testing.assert_equal(self.data_true, rf['data'][:])
        rf.close()

    def test_get(self):
        pass

    def test_set(self):
        pass

    def test_delete(self):
        pass

    def test_getitem(self):
        pass

    def test_arithmetic(self):
        pass


class TestBrainEncoder:

    # Prepare brain activation
    n_sample = 30
    n_meas = 2
    brain_activ = np.random.randn(n_sample, n_meas)

    def test_encode_dnn(self):

        # prepare dnn activation
        cv = 2
        dnn_activ = Activation()
        dnn_activ.set('conv5', np.random.randn(self.n_sample, 1, 3, 3))
        dnn_activ.set('fc3', np.random.randn(self.n_sample, 10, 1, 1))
        encoder = BrainEncoder(self.brain_activ, 'uv', 'corr', cv)

        # test uv/corr and iter_axis=None
        encode_dict = encoder.encode_dnn(dnn_activ)
        assert list(encode_dict.keys()) == dnn_activ.layers
        v1_keys = sorted(['max_score', 'max_loc'])
        for k1, v1 in encode_dict.items():
            assert sorted(v1.keys()) == v1_keys
            assert v1['max_score'].shape == (1, self.n_meas)
            if k1 == 'conv5':
                assert np.all(v1['max_loc'][..., 0] == 1)
            elif k1 == 'fc3':
                assert np.all(v1['max_loc'][..., 1] == 1)
                assert np.all(v1['max_loc'][..., 2] == 1)

        # test mv/glm and iter_axis=channel
        encoder.set(model_type='mv', model_name='glm', cv=cv)
        encode_dict = encoder.encode_dnn(dnn_activ, 'channel')
        assert list(encode_dict.keys()) == dnn_activ.layers
        v1_keys = sorted(['score', 'model'])
        for k1, v1 in encode_dict.items():
            assert sorted(v1.keys()) == v1_keys
            n_chn = dnn_activ.get(k1).shape[1]
            assert v1['score'].shape == (n_chn, self.n_meas, cv)
            assert v1['model'].shape == (n_chn, self.n_meas)

    def test_encode_behavior(self):

        # prepare behavior data
        beh_data = np.random.randn(10, 1)

        # test uv
        encoder = BrainEncoder(self.brain_activ, 'uv', 'lasso')
        pred_dict = encoder.encode_behavior(beh_data)
        assert sorted(pred_dict.keys()) == sorted(['score', 'model', 'location'])
        assert np.all(pred_dict['location'] == 0)
        for v in pred_dict.values():
            assert v.shape == (self.brain_activ.shape[1],)

        # test mv
        encoder.set(model_type='mv', model_name='lasso')
        pred_dict = encoder.encode_behavior(beh_data)
        assert sorted(pred_dict.keys()) == sorted(['score', 'model'])
        for v in pred_dict.values():
            assert v.shape == (self.brain_activ.shape[1],)


class TestBrainDecoder:

    # Prepare brain activation
    brain_activ = np.random.randn(10, 2)

    def test_decode_dnn(self):

        # prepare dnn activation
        dnn_activ = Activation()
        dnn_activ.set('conv5', np.random.randn(10, 2, 3, 3))
        dnn_activ.set('fc3', np.random.randn(10, 10, 1, 1))

        # test uv
        decoder = BrainDecoder(self.brain_activ, 'uv', 'glm')
        pred_dict = decoder.decode_dnn(dnn_activ)
        assert list(pred_dict.keys()) == dnn_activ.layers
        v1_keys = sorted(['score', 'model', 'location'])
        for k1, v1 in pred_dict.items():
            assert sorted(v1.keys()) == v1_keys
            _, n_chn, n_row, n_col = dnn_activ.get(k1).shape
            for v2 in v1.values():
                assert v2.shape == (n_chn, n_row, n_col)

        # test mv
        decoder.set(model_type='mv', model_name='glm')
        pred_dict = decoder.decode_dnn(dnn_activ)
        assert list(pred_dict.keys()) == dnn_activ.layers
        v1_keys = sorted(['score', 'model'])
        for k1, v1 in pred_dict.items():
            assert sorted(v1.keys()) == v1_keys
            _, n_chn, n_row, n_col = dnn_activ.get(k1).shape
            for v2 in v1.values():
                assert v2.shape == (n_chn, n_row, n_col)

    def test_decode_behavior(self):

        # prepare behavior data
        beh_data = np.random.randint(1, 3, (10, 1))

        # test uv
        decoder = BrainDecoder(self.brain_activ, 'uv', 'lrc')
        pred_dict = decoder.decode_behavior(beh_data)
        assert sorted(pred_dict.keys()) == sorted(['score', 'model', 'location'])
        for v in pred_dict.values():
            assert v.shape == (beh_data.shape[1],)

        # test mv
        decoder.set(model_type='mv', model_name='lrc')
        pred_dict = decoder.decode_behavior(beh_data)
        assert sorted(pred_dict.keys()) == sorted(['score', 'model'])
        for v in pred_dict.values():
            assert v.shape == (beh_data.shape[1],)


if __name__ == '__main__':
    pytest.main()
