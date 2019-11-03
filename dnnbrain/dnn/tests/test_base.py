import os
import cv2
import unittest
import torch
import numpy as np
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

    # test frame_nums
    def test_init(self):
        vid_file = pjoin(DNNBRAIN_TEST, 'video', \
                         'sub-CSI1_ses-01_imagenet.mp4')
        frame_nums = list(np.random.randint(0,148,20))
        dataset = VideoSet(vid_file, frame_nums)
        self.assertEqual(dataset.frame_nums,frame_nums)
        
    # test video in each frames
    def test_getitem(self):  
        # test slice   
        vid_file = pjoin(DNNBRAIN_TEST, 'video', \
                         'sub-CSI1_ses-01_imagenet.mp4')
        dataset = VideoSet(vid_file, list(range(148)))
        transform = transforms.Compose([transforms.ToTensor()])
        indices = slice(1,4)
        tmpvi,_ = dataset[indices]
        for i in list(range(148))[indices]:
            tmp_video, _ = dataset[i]
            cap = cv2.VideoCapture(vid_file)
            for j in range(i):
                _, tmp_video1 = cap.read()
            frame = Image.fromarray(cv2.cvtColor(tmp_video1, cv2.COLOR_BGR2RGB))
            tmp_video1 = transform(frame)
            self.assertTrue(torch.equal(tmp_video,tmp_video1))
        
        # test index
        vid_file = pjoin(DNNBRAIN_TEST, 'video', \
                         'sub-CSI1_ses-01_imagenet.mp4')
        dataset = VideoSet(vid_file, list(range(148)))
        transform = transforms.Compose([transforms.ToTensor()])
        indices = list(np.random.randint(0,148,20))
        tmpvi,_ = dataset[indices]
        for i in range(len(indices)):
            tmp_video = tmpvi[i]
            cap = cv2.VideoCapture(vid_file)
            for j in range(indices[i]):
                _, tmp_video2 = cap.read()
            frame = Image.fromarray(cv2.cvtColor(tmp_video2, cv2.COLOR_BGR2RGB))
            tmp_video2 = transform(frame)
            self.assertTrue(torch.equal(tmp_video,tmp_video2))
        
        # test int
        vid_file = pjoin(DNNBRAIN_TEST, 'video', \
                         'sub-CSI1_ses-01_imagenet.mp4')
        frame_nums = list(np.random.randint(0,148,20))
        dataset = VideoSet(vid_file, frame_nums)
        transform = transforms.Compose([transforms.ToTensor()])
        for i in range(len(frame_nums)):
            tmp_video, _ = dataset[i]
            cap = cv2.VideoCapture(vid_file)
            for j in range(frame_nums[i]):
                _, tmp_video3 = cap.read()
            frame = Image.fromarray(cv2.cvtColor(tmp_video3, cv2.COLOR_BGR2RGB))
            tmp_video3 = transform(frame)
            self.assertTrue(torch.equal(tmp_video,tmp_video3))    


if __name__ == '__main__':
    unittest.main()
