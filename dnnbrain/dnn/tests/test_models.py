import os
import h5py
import pytest
import numpy as np

from PIL import Image
from os.path import join as pjoin
from dnnbrain.dnn import core as dcore
from dnnbrain.dnn import models as db_models

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestAlexNet:

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
        dnn = db_models.AlexNet()
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

        # compute with pool
        activation3 = dnn.compute_activation(stimuli1, dmask, 'max')
        activation4 = activation1.pool('max')

        # assert
        np.testing.assert_almost_equal(np.asarray(rf['conv5']),
                                       activation1.get('conv5'), 4)
        np.testing.assert_almost_equal(np.asarray(rf['fc3']),
                                       activation1.get('fc3'), 4)
        np.testing.assert_equal(activation1.get('conv5'), activation2.get('conv5'))
        np.testing.assert_equal(activation1.get('fc3'), activation2.get('fc3'))
        np.testing.assert_equal(activation3.get('conv5'), activation4.get('conv5'))
        np.testing.assert_equal(activation3.get('fc3'), activation4.get('fc3'))

        rf.close()

    def test_get_kernel(self):
        pass

    def test_ablate(self):
        pass


@pytest.mark.skip
def test_dnn_truncate():
    """
    Test dnn_truncate
    """
    pass
    

@pytest.mark.skip
def test_dnn_train_model():
    """
    Test dnn_train_model
    """
    pass
    
    
@pytest.mark.skip
def test_dnn_test_model():
    """
    Test dnn_test_model
    """
    pass


if __name__ == '__main__':
    pytest.main()
