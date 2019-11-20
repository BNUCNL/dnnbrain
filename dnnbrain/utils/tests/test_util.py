import os
import pytest
import numpy as np

from os.path import join as pjoin
from dnnbrain.utils import util as db_util

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


def test_gen_dmask():

    # prepare
    layers1 = ['conv1']
    layers2 = ['conv1', 'conv2']
    channels1 = [1]
    channels2 = [1, 2]
    dmask_file = pjoin(DNNBRAIN_TEST, 'alexnet.dmask.csv')

    # -assert parameters-
    with pytest.raises(AssertionError):
        db_util.gen_dmask()
    with pytest.raises(AssertionError):
        db_util.gen_dmask(layers1, dmask_file=dmask_file)
    with pytest.raises(AssertionError):
        db_util.gen_dmask(channels=channels1, dmask_file=dmask_file)

    # -assert single layer-
    # assert with channels
    dmask = db_util.gen_dmask(layers1, channels2)
    assert dmask.layers == layers1
    assert dmask.get(layers1[0])['chn'] == channels2

    # assert without channels
    dmask = db_util.gen_dmask(layers1)
    assert dmask.layers == layers1
    assert dmask.get(layers1[0]).get('chn') is None

    # -assert multi layers-
    # assert with channels
    dmask = db_util.gen_dmask(layers2, channels2)
    assert dmask.layers == layers2
    for layer, chn in zip(layers2, channels2):
        assert dmask.get(layer)['chn'] == [chn]
    with pytest.raises(Exception):
        db_util.gen_dmask(layers2, channels1)

    # assert without channels
    dmask = db_util.gen_dmask(layers2)
    assert dmask.layers == layers2
    for layer in layers2:
        assert dmask.get(layer).get('chn') is None

    # -assert dmask file-
    dmask = db_util.gen_dmask(dmask_file=dmask_file)
    assert dmask.layers == ['conv5', 'fc3']
    assert dmask.get('conv5')['chn'] == [1, 2, 3]
    assert dmask.get('conv5')['row'] == [4, 5]
    assert dmask.get('conv5')['col'] == [6, 7, 8]
    assert dmask.get('fc3')['chn'] == [1, 2, 3]


def test_normalize():

    # prepare
    arr = np.random.randn(2, 3)
    arr_true = (arr - arr.min()) / (arr.max() - arr.min())
    arr_norm = db_util.normalize(arr)

    # assert
    assert arr_norm.min() == 0
    assert arr_norm.max() == 1
    np.testing.assert_equal(arr_norm, arr_true)
