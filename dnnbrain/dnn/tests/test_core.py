import os
import pytest

from os.path import join as pjoin

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestStimulus:

    def test_load(self):
        pass

    def test_save(self):
        pass

    def test_get(self):
        pass

    def test_set(self):
        pass

    def test_delete(self):
        pass

    def test_getitem(self):
        pass


class TestDNN:

    def test_load(self):
        pass

    def test_save(self):
        pass

    def test_compute_activation(self):
        pass

    def test_get_kernel(self):
        pass

    def test_ablate(self):
        pass


class TestActivation:

    def test_load(self):
        pass

    def test_save(self):
        pass

    def test_get(self):
        pass

    def test_set(self):
        pass

    def test_delete(self):
        pass

    def test_concatenate(self):
        pass

    def test_mask(self):
        pass

    def test_pool(self):
        pass

    def test_fe(self):
        pass

    def test_arithmetic(self):
        pass


class TestMask:

    def test_load(self):
        pass

    def test_save(self):
        pass

    def test_get(self):
        pass

    def test_set(self):
        pass

    def test_delete(self):
        pass


if __name__ == '__main__':
    pytest.main()
