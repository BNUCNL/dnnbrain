import os
import unittest

from os.path import join as pjoin
from dnnbrain.io.file import StimulusFile
DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

stim_file=pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.stim.csv')
class TestStimulusFile(unittest.TestCase):

	def test_read(self):
		act=StimulusFile(stim_file).read()
		TYPE='image'
		PATH='/nfs/s2/dnnbrain_data/test/image/images'
		TITLE='ImageNet images in all 5000scenes runs of sub-CSI1_ses-01'
		ID=['n01930112_19568.JPEG','n03733281_29214.JPEG']
		RT=[3.6309,4.2031]
		self.assertEqual(act['type'],TYPE)
		self.assertEqual(act['title'],TITLE)
		self.assertEqual(act['path'],PATH)
		self.assertEqual(act['data']['stimID'][0:1],ID[0:1])
		self.assertEqual(act['data']['RT'][0:1],RT[0:1])

if __name__=='__main__':
	unittest.main()

	def test_write(self):
	        pass


class TestActivationFile(unittest.TestCase):

	def test_read(self):
	        pass

	def test_write(self):
	        pass


class TestMaskFile(unittest.TestCase):

	def test_read(self):
		pass

	def test_write(self):
	        pass
