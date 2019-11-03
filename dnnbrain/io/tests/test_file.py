import os
import unittest
import numpy as np
import h5py

from dnnbrain.io.file import ActivationFile
from os.path import join as pjoin
from dnnbrain.io.file import StimulusFile

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
stim_file=pjoin(DNNBRAIN_TEST,'image','sub-CSI1_ses-01_imagenet.stim.csv')

if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)



class TestStimulusFile(unittest.TestCase):

	def test_read(self):
        
		stim=StimulusFile(stim_file).read()
		stim_type='image'
		stim_pyth='/nfs/s2/dnnbrain_data/test/image/images'
		stim_title='ImageNet images in all 5000scenes runs of sub-CSI1_ses-01'
		stim_id=['n01930112_19568.JPEG','n03733281_29214.JPEG']
		stim_rt=[3.6309,4.2031]
		self.assertEqual(stim['type'],stim_type)
		self.assertEqual(stim['title'],stim_title)
		self.assertEqual(stim['path'],stim_pyth)
		self.assertEqual(stim['data']['stimID'][0:1],stim_id[0:1])
		self.assertEqual(stim['data']['RT'][0:1],stim_rt[0:1])

	def test_write(self):
	        pass


class TestActivationFile(unittest.TestCase):

	def test_read(self):
	        pass

<<<<<<< HEAD
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
            
=======
	def test_write(self):
	        pass

>>>>>>> 07ed91df53191a316d48684b31d758aea04dce0d

class TestMaskFile(unittest.TestCase):

	def test_read(self):
		pass

	def test_write(self):
	        pass

if __name__=='__main__':
	unittest.main()

<<<<<<< HEAD
    def test_write(self):
        pass


if __name__=='__main__':
    unittest.main()
=======
>>>>>>> 07ed91df53191a316d48684b31d758aea04dce0d
