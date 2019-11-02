import os
import unittest

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
        pass


class TestMaskFile(unittest.TestCase):

    def test_read(self):
        pass

    def test_write(self):
        pass
