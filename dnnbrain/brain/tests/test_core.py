import os
import h5py
import pytest
import numpy as np

from os.path import join as pjoin
from dnnbrain.brain.core import ROI

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
        roi = ROI(fname)
        assert roi.rois == rf.attrs['roi'].tolist()
        np.testing.assert_equal(roi.data, rf['data'][:])

        # assert read with rois
        roi = ROI(fname, 'PHA1_R')
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


if __name__ == '__main__':
    pytest.main()
