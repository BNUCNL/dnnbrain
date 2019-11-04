import os
import cv2
import torch
import pytest
import numpy as np

from PIL import Image
from os.path import join as pjoin
from torchvision import transforms
from dnnbrain.dnn.base import ImageSet, VideoSet


DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestImageSet:

    def test_init(self):

        # test when labels and transform are None
        img_dir = pjoin(DNNBRAIN_TEST, 'image', 'images')
        img_ids = ['n01443537_2819.JPEG', 'n01531178_2651.JPEG']

        dataset = ImageSet(img_dir, img_ids)

        # test dir ids labels
        assert dataset.img_dir == img_dir
        assert dataset.img_ids == ['n01443537_2819.JPEG', 'n01531178_2651.JPEG']
        assert np.all(dataset.labels == np.array([1, 1]))
        
        # test transform
        for img_id in dataset.img_ids:
            image = Image.open(pjoin(dataset.img_dir, img_id))  # load image
            image_org = dataset.transform(image)               
            image_new = transforms.Compose([transforms.ToTensor()])(image)
            assert torch.equal(image_org, image_new)   # compare original image and new image
        
        # test when labels and transform are given
        labels = [1, 11]
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])
    
        dataset = ImageSet(img_dir, img_ids, labels, transform)
        
        assert np.all(dataset.labels == np.array([1, 11]))
        
        for img_id in dataset.img_ids:
            image = Image.open(pjoin(dataset.img_dir, img_id))  # load image
            image_org = dataset.transform(image)
            image_new = transform(image)
            assert torch.equal(image_org, image_new)
    
    def test_getitem(self):
        
        # initialize the dir ids labels transform & dataset
        img_dir = pjoin(DNNBRAIN_TEST, 'image', 'images')
        img_ids = ['n01443537_2819.JPEG', 'n01531178_2651.JPEG',
                   'n07695742_5848.JPEG', 'n02655020_1972.JPEG', 'n01641577_1229.JPEG']
        labels = [1, 11, 32, 397, 30]
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])
        
        dataset = ImageSet(img_dir, img_ids, labels, transform)
        
        # test indice int
        indices = 3
        image_org, labels_get = dataset[indices]
        
        image_new = transform(Image.open(pjoin(img_dir, img_ids[indices])))

        assert torch.equal(image_org, image_new)
        assert labels_get[0] == labels[indices]
        
        # test indice list
        indices = [1, 2]
        image_org, labels_get = dataset[indices]
        
        tmp_ids = [dataset.img_ids[i] for i in indices]
        image_new = torch.zeros(0)
        for img_id in tmp_ids:
            img_tmp = transform(Image.open(pjoin(img_dir, img_id)))
            img_tmp = torch.unsqueeze(img_tmp, 0)
            image_new = torch.cat((image_new, img_tmp))

        assert torch.equal(image_org, image_new)
        assert labels_get == [labels[i] for i in indices]
        
        # test indice slice
        indices = slice(1, 3)
        image_org, labels_get = dataset[indices]
        
        tmp_ids = dataset.img_ids[indices]
        image_new = torch.zeros(0)
        for img_id in tmp_ids:
            img_tmp = transform(Image.open(pjoin(img_dir, img_id)))
            img_tmp = torch.unsqueeze(img_tmp, 0)
            image_new = torch.cat((image_new, img_tmp))

        assert torch.equal(image_org, image_new)
        assert labels_get, labels[indices]


class TestVideoSet:

    def test_init(self):
        vid_file = pjoin(DNNBRAIN_TEST, 'video', 'sub-CSI1_ses-01_imagenet.mp4')
        frame_nums = list(np.random.randint(0, 148, 20))
        dataset = VideoSet(vid_file, frame_nums)
        assert dataset.frame_nums == frame_nums

    # test video in each frames
    def test_getitem(self):

        # test slice
        vid_file = pjoin(DNNBRAIN_TEST, 'video', 'sub-CSI1_ses-01_imagenet.mp4')
        transform = transforms.Compose([transforms.ToTensor()])
        frame_list = [4, 6, 2, 8, 54, 23, 127]
        dataset = VideoSet(vid_file, frame_list)
        indices = slice(0, 5)
        tmpvi, _ = dataset[indices]
        for ii, i in enumerate(frame_list[indices]):
            cap = cv2.VideoCapture(vid_file)
            for j in range(i):
                _, tmp = cap.read()
            frame = Image.fromarray(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
            tmp = transform(frame)
            assert torch.equal(tmp, tmpvi[ii])
            
        # test int
        vid_file = pjoin(DNNBRAIN_TEST, 'video', 'sub-CSI1_ses-01_imagenet.mp4')
        frame_nums = list(np.random.randint(0, 148, 20))
        dataset = VideoSet(vid_file, frame_nums)
        transform = transforms.Compose([transforms.ToTensor()])
        for i in range(len(frame_nums)):
            tmp_video, _ = dataset[i]
            cap = cv2.VideoCapture(vid_file)
            for j in range(frame_nums[i]):
                _, tmp_video3 = cap.read()
            frame = Image.fromarray(cv2.cvtColor(tmp_video3, cv2.COLOR_BGR2RGB))
            tmp_video3 = transform(frame)
            assert torch.equal(tmp_video, tmp_video3)
            
        # test list
        vid_file = pjoin(DNNBRAIN_TEST, 'video', 'sub-CSI1_ses-01_imagenet.mp4')
        transform = transforms.Compose([transforms.ToTensor()])
        frame_list = [2, 8, 54, 127, 128, 129, 130]
        dataset = VideoSet(vid_file, frame_list)
        indices = [1, 2, 4, 5]
        tmpvi, _ = dataset[indices]
        for ii, i in enumerate([frame_list[i] for i in indices]):
            cap = cv2.VideoCapture(vid_file)
            for j in range(0, i):
                _, tmp = cap.read()
            frame = Image.fromarray(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
            tmp = transform(frame)
            assert torch.equal(tmp, tmpvi[ii])


if __name__ == '__main__':
    pytest.main()
