import dnnbrain
import os
import pytest

# Define dnnbrain_path as a global variable in pytest
@pytest.fixture
def dnnbrain_path():
    return os.path.dirname(dnnbrain.__file__)