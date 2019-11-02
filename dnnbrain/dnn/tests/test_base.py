import os
import cv2
import unittest
import numpy as np

from PIL import Image
from os.path import join as pjoin
from dnnbrain.dnn.base import VideoSet
from torchvision import transforms

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
        # prepare variates
        fpath = pjoin(DNNBRAIN_TEST, 'video', 'sub-CSI1_ses-01_imagenet.mp4')
        frame_id = [1, 2, 4]
        print(frame_id)

        # initiate
        video_1 = VideoSet(fpath, frame_id)
        self.assertEqual(video_1.frame_nums, frame_id)

    def test_getitem(self):
        # prepare variates
        fpath = pjoin(DNNBRAIN_TEST, 'video', 'sub-CSI1_ses-01_imagenet.mp4')
        frame_id = [1, 2, 4]

        # initiate
        video_1 = VideoSet(fpath, frame_id)
        transform = transforms.Compose([transforms.ToTensor()])

        for i in list(range(len(frame_id))):
            video_test_1, _ = video_1[i]
            vid_cap = cv2.VideoCapture(fpath)
            for j in range(frame_id[i]):
                _, video_test_2 = vid_cap.read()
            video_test_2 = Image.fromarray(cv2.cvtColor(video_test_2, cv2.COLOR_BGR2RGB))
            video_test_2 = transform(video_test_2)
            self.assertTrue(np.all(video_test_1.numpy() == video_test_2.numpy()))


if __name__ == '__main__':
    unittest.main()
