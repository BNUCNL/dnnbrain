import os
import unittest

from os.path import join as pjoin
from torchvision import transforms
from dnnbrain.dnn.base import ImageSet

import numpy as np
from PIL import Image
import torch

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestImageSet(unittest.TestCase):

    def test_init(self):

        #test when labels and transform are None
        img_dir = pjoin(DNNBRAIN_TEST, 'image', 'images')
        img_ids = ['n01443537_2819.JPEG', 'n01531178_2651.JPEG']

        dataset = ImageSet(img_dir,img_ids)

        self.assertEqual(dataset.img_dir,'/nfs/s2/dnnbrain_data/test/image/images')
        self.assertEqual(dataset.img_ids, ['n01443537_2819.JPEG', 'n01531178_2651.JPEG'])
        self.assertTrue(np.all(dataset.labels == np.array([1,1])))

        for img_id in dataset.img_ids:
            image = Image.open(pjoin(dataset.img_dir, img_id))  # load image
            image_a = dataset.transform(image)
            image_b = transforms.Compose([transforms.ToTensor()])(image)
            self.assertTrue(torch.equal(image_a,image_b))
        
        #test when labels and transform are given
        labels = [1, 11]
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
    
        dataset = ImageSet(img_dir,img_ids,labels,transform)
        
        self.assertTrue(np.all(dataset.labels == np.array([1,11])))  
        
        for img_id in dataset.img_ids:
            image = Image.open(pjoin(dataset.img_dir, img_id))  # load image
            image_a = dataset.transform(image)
            image_b = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])(image)
            self.assertTrue(torch.equal(image_a,image_b))
    
    def test_getitem(self):
        
        img_par = pjoin(DNNBRAIN_TEST, 'image', 'images')
        img_ids = ['n01443537_2819.JPEG', 'n01531178_2651.JPEG','n07695742_5848.JPEG','n02655020_1972.JPEG','n01641577_1229.JPEG']

        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()])
        labels = [1,11,32,397,30]
        
        dataset = ImageSet(img_par,img_ids,labels,transform)
        
        #test indice int
        indices = 3
        data,labels_get = dataset.__getitem__(indices)
        
        img_org = dataset[indices]
        img0 = img_org[0]
        img_0 = transform(Image.open(pjoin(img_par, img_ids[indices])))

        self.assertTrue(np.all(img0.numpy() == img_0.numpy()))
        self.assertEqual(labels_get[0],labels[indices])
        
        #test indice list
        indices = [1,2]
        data,labels_get = dataset.__getitem__(indices)
        
        img_org = dataset[indices]
        img0 = img_org[0]
        
        tmp_ids = [dataset.img_ids[i] for i in indices]
        img_0 = torch.zeros(0)
        for img_id in tmp_ids:
            img_tmp = transform(Image.open(pjoin(img_par, img_id)))
            img_tmp = torch.unsqueeze(img_tmp, 0)
            img_0 = torch.cat((img_0, img_tmp))

        self.assertTrue(np.all(img0.numpy() == img_0.numpy()))
        self.assertEqual(labels_get,[labels[i] for i in indices])
        
        #test indice slice
        indices = slice(1,3)
        data,labels_get = dataset.__getitem__(indices)

        img_org = dataset[indices]  
        img0 = img_org[0]
        
        tmp_ids = dataset.img_ids[indices]
        img_0 = torch.zeros(0)
        for img_id in tmp_ids:
            img_tmp = transform(Image.open(pjoin(img_par, img_id)))
            img_tmp = torch.unsqueeze(img_tmp, 0)
            img_0 = torch.cat((img_0, img_tmp))

        self.assertTrue(np.all(img0.numpy() == img_0.numpy()))
        self.assertEqual(labels_get,labels[indices])               
        
class TestVideoSet(unittest.TestCase):

    def test_init(self):
        pass

    def test_getitem(self):
        pass


if __name__ == '__main__':
    
    unittest.main()