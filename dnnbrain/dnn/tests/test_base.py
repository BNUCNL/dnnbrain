import os
import cv2
import unittest
import torch
from PIL import Image
from dnn.base import VideoSet
from torchvision import transforms
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
        vid_file = pjoin(DNNBRAIN_TEST, 'video', \
                         'sub-CSI1_ses-01_imagenet.mp4')
        frame_nums = [12,13,15,16]
        dataset = VideoSet(vid_file, frame_nums)
        transform = transforms.Compose([transforms.ToTensor()])
        
        for i in list(range(len(frame_nums))):
            tmp_video1, _ = dataset[i]
            cap = cv2.VideoCapture(vid_file)
            for j in range(frame_nums[i]):
                _, tmp_video2 = cap.read()
            frame = Image.fromarray(cv2.cvtColor(tmp_video2, cv2.COLOR_BGR2RGB))
            tmp_video2 = transform(frame)
            ist = torch.equal(tmp_video1,tmp_video2)
            print(ist)
