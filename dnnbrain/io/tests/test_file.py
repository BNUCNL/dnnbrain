import os
import unittest
import numpy as np
import h5py

from dnnbrain.io.file import ActivationFile
from os.path import join as pjoin

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestStimulusFile(unittest.TestCase):

    def test_read(self):
        pass

    def test_write(self):
        pass


class TestActivationFile(unittest.TestCase):

    def test_read(self):
        pass

    def test_write(self):        
        fpath = pjoin(TMP_DIR,'test.act.h5')
        act = {}
        layers = list(range(10))
        for i in range(10):
            layers[i] = f"layers{i}"
        for i in range(10):
            act.update({layers[i]: {'data':np.random.randn(2,3),'raw_shape':(2,3)}})
        
        # write
        ActivationFile(fpath).write(act)
                
        # compare
        rf = h5py.File(fpath,'r')
        for i in range(10):
            self.assertTrue(np.all(act[layers[i]]['data'] == np.array(rf[layers[i]])))
            self.assertTrue(np.all(np.array(act[layers[i]]['raw_shape']) == rf[layers[i]].attrs['raw_shape']))
            

class TestMaskFile(unittest.TestCase):

    def test_read(self):
        pass

    def test_write(self):
        pass


if __name__=='__main__':
    unittest.main()