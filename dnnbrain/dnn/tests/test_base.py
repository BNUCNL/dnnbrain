import os
import unittest

from os.path import join as pjoin

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestImageSet(unittest.TestCase):

    def test_init(self):
        pass

    def test_getitem(self):
        pass


class TestVideoSet(unittest.TestCase):

    def test_init(self):
        pass

    def test_getitem(self):
        pass
