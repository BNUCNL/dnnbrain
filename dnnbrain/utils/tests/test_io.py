import os
import unittest
import numpy as np

from os.path import join as pjoin
from dnnbrain.utils import io as uio

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')


class TestStimCsvIO(unittest.TestCase):

    def test_read_stim_csv(self):
        fpath = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.stim.csv')
        stim_dict = uio.read_stim_csv(fpath)

        self.assertEqual(stim_dict['type'], 'image')
        self.assertEqual(list(stim_dict['stim'].keys()),
                         ['stimID', 'condition', 'onset', 'duration'])
        self.assertEqual(list(stim_dict['meas'].keys()),
                         ['response', 'RT'])
        self.assertEqual(stim_dict['stim']['stimID'][[0, 3, -1]].tolist(),
                         ['n01930112_19568.JPEG', 'n01917289_1429.JPEG', 'n04557648_10553.JPEG'])
        self.assertEqual(stim_dict['meas']['RT'][[0, 3, -1]].tolist(),
                         [3.6309, 3.622, 2.806])

    def test_save_stim_csv(self):
        # prepare
        fpath = pjoin(TMP_DIR, 'test.stim.csv')
        title = 'test title'
        type = 'video'
        path = 'path'
        stim_var_dict = {
            'stimID': [1, 2, 3],
            'label': [0, 1, 2]
        }
        meas_var_dict = {'RT': [0.1, 0.2, 0.3]}
        opt_meta = {'test': 'opt_meta'}
        uio.save_stim_csv(fpath, title, type, path, stim_var_dict, meas_var_dict, opt_meta)

        # assert
        stim_dict = uio.read_stim_csv(fpath)
        self.assertEqual(stim_dict['title'], title)
        self.assertEqual(stim_dict['type'], type)
        self.assertEqual(stim_dict['path'], path)
        self.assertEqual(stim_dict['stim']['stimID'].tolist(), stim_var_dict['stimID'])
        self.assertEqual(stim_dict['stim']['label'].tolist(), stim_var_dict['label'])
        self.assertEqual(stim_dict['meas']['RT'].tolist(), meas_var_dict['RT'])
        self.assertEqual(stim_dict['test'], opt_meta['test'])


if __name__ == '__main__':
    unittest.main()
