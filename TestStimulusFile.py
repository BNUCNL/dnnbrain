import os
import numpy as np
from os.path import join as pjoin
from dnnbrain.io.file import StimulusFile
import unittest

DNNBRAIN_TEST =pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
stim_file = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.stim.csv')
new_stim_file = pjoin(TMP_DIR, 'write_check.stim.csv')


class TestStimulusFile(unittest.TestCase):
	# initial varible put in the __init__ funciton instead of the class, which means succession.
	def write_check(self):
		TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
		newfile = pjoin(TMP_DIR, 'write_check.stim.csv')
		test_sti = dict(
			# when type = 'image', the list of stiID should be composed of string.
			type='image', 
			path='dnnbrain/data/test_data/picture_folder/ants',
			title='great_data',
			data=dict(
				stimID=np.array(['1', '2', '3']),
				label=np.array([1, 2, 3]),
				acc=np.array([1, 2, 3]),
				RT=np.array([1, 2, 3])
				) 
			)
		StimulusFile(newfile).write(
			type=test_sti['type'], 
			stim_path=test_sti['path'], 
			data=test_sti['data'], 
			title=test_sti['title']
			)
		check_sti = StimulusFile(newfile).read()
			
		self.assertEqual(test_sti['type'], check_sti['type'])
		self.assertEqual(test_sti['path'], check_sti['path'])
		self.assertEqual(test_sti['title'], check_sti['title'])
		self.assertTrue(np.all(test_sti['data']['stimID'] == np.array(check_sti['data']['stimID'])))
		self.assertTrue(np.all(test_sti['data']['label'] == np.array(check_sti['data']['label'])))
		self.assertTrue(np.all(test_sti['data']['acc'] == np.array(check_sti['data']['acc'])))
		self.assertTrue(np.all(test_sti['data']['RT'] == np.array(check_sti['data']['RT'])))


test_file = TestStimulusFile()
test_file.write_check()

if __name__ == '__main__':
	unittest.main()
