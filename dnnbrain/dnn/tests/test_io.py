import pytest
from dnnbrain.dnn import io as iofiles
import os

@pytest.fixture()
def dnnbrain_path():
    pythonpath_list = os.environ['PYTHONPATH'].split(os.pathsep)
    try:
        dnnbrain_idx = [i for i,pl in enumerate(pythonpath_list) if 'dnnbrain' in pl][0]
    except IndexError:
        raise Exception('Please set dnnbrain into your PYTHONPATH environment.')
    return pythonpath_list[dnnbrain_idx]


def test_read_Imagefolder_output(dnnbrain_path):
    """
    Test output of function: read_Imagefolder.
    """
    # dnnbrain_path
    
    parpath = os.path.join(dnnbrain_path, 'dnnbrain', 'data', 'test_data', 'picture_folder')
    picpath, conditions = iofiles.read_Imagefolder(parpath)
    # output type
    assert isinstance(picpath, list)
    assert isinstance(conditions, list)
    # output correction
    picname_actual = ['ant1.jpg', 'ant2.jpg', 'bees1.jpg', 'bees2.jpg']
    conditions_actual = ['ants', 'ants', 'bees', 'bees']
    picpath_actual = [os.path.join(parpath, conditions[i], picname_actual[i]) for i, _ in enumerate(picname_actual)]
    assert conditions == conditions_actual
    assert picpath == picpath_actual
    