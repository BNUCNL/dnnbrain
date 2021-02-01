import os
import cv2
import copy
import torch
import pytest
import numpy as np

from PIL import Image
from os.path import join as pjoin
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score
from torchvision import transforms
from dnnbrain.dnn import base as db_base
import matplotlib.pyplot as plt

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestImageSet:

    def test_init(self):

        # test when labels and transform are None
        img_dir = pjoin(DNNBRAIN_TEST, 'image', 'images')
        img_ids = ['n01443537_2819.JPEG', 'n01531178_2651.JPEG']

        dataset = db_base.ImageSet(img_dir, img_ids)

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
    
        dataset = db_base.ImageSet(img_dir, img_ids, labels, transform)
        
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
        
        dataset = db_base.ImageSet(img_dir, img_ids, labels, transform)
        
        # test indice int
        indices = 3
        image_org, labels_get = dataset[indices]
        
        image_new = transform(Image.open(pjoin(img_dir, img_ids[indices])))

        assert torch.equal(image_org, image_new)
        assert labels_get == labels[indices]
        
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
        np.testing.assert_equal(labels_get, labels[indices])


class TestVideoSet:

    def test_init(self):
        vid_file = pjoin(DNNBRAIN_TEST, 'video', 'sub-CSI1_ses-01_imagenet.mp4')
        frame_nums = list(np.random.randint(0, 148, 20))
        dataset = db_base.VideoSet(vid_file, frame_nums)
        assert dataset.frame_nums == frame_nums

    # test video in each frames
    def test_getitem(self):

        # test slice
        vid_file = pjoin(DNNBRAIN_TEST, 'video', 'sub-CSI1_ses-01_imagenet.mp4')
        transform = transforms.Compose([transforms.ToTensor()])
        frame_list = [4, 6, 2, 8, 54, 23, 127]
        dataset = db_base.VideoSet(vid_file, frame_list)
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
        dataset = db_base.VideoSet(vid_file, frame_nums)
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
        dataset = db_base.VideoSet(vid_file, frame_list)
        indices = [1, 2, 4, 5]
        tmpvi, _ = dataset[indices]
        for ii, i in enumerate([frame_list[i] for i in indices]):
            cap = cv2.VideoCapture(vid_file)
            for j in range(0, i):
                _, tmp = cap.read()
            frame = Image.fromarray(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
            tmp = transform(frame)
            assert torch.equal(tmp, tmpvi[ii])


class TestVideoClipSet:

    # test video in each frames
    def test_getitem(self):

        frame_size = (112, 112)

        # test int
        clip_files = [pjoin(DNNBRAIN_TEST, 'video', 'sub-CSI1_ses-01_imagenet.mp4')]
        transform = transforms.Compose([transforms.Resize(frame_size),
                                        transforms.ToTensor()])
        dataset = db_base.VideoClipSet(clip_files, transform)
        assert len(dataset) == 1

        for idx, file in enumerate(clip_files):
            data, labels = dataset[idx]
            cap = cv2.VideoCapture(file)
            n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert data.shape == (3, n_frame, *frame_size)
            assert labels == 0
            for i in range(n_frame):
                _, frame = cap.read()
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = transform(frame)
                assert torch.equal(data[:, i], frame)


class TestImageProcessor:

    image = np.random.randint(0, 256, (3, 5, 5), np.uint8)
    ip = db_base.ImageProcessor()

    def test_to_array(self):

        # test RGB array
        image_out = self.ip.to_array(self.image)
        assert isinstance(image_out, np.ndarray)
        np.testing.assert_equal(self.image, image_out)

        # test gray array
        image_out = self.ip.to_array(self.image[0])
        assert isinstance(image_out, np.ndarray)
        np.testing.assert_equal(self.image[0], image_out)

        # test RGB tensor
        tensor = torch.from_numpy(self.image)
        image_out = self.ip.to_array(tensor)
        assert isinstance(image_out, np.ndarray)
        np.testing.assert_equal(self.image, image_out)

        # test gray tensor
        tensor = torch.from_numpy(self.image[0])
        image_out = self.ip.to_array(tensor)
        assert isinstance(image_out, np.ndarray)
        np.testing.assert_equal(self.image[0], image_out)

        # test RGB PIL.Image
        pil = Image.fromarray(self.image.transpose((1, 2, 0)))
        image_out = self.ip.to_array(pil)
        assert isinstance(image_out, np.ndarray)
        np.testing.assert_equal(self.image, image_out)

        # test gray PIL.Image
        pil = Image.fromarray(self.image[0])
        image_out = self.ip.to_array(pil)
        assert isinstance(image_out, np.ndarray)
        np.testing.assert_equal(self.image[0], image_out)

    def test_to_tensor(self):

        # test RGB array
        image_out = self.ip.to_tensor(self.image)
        assert isinstance(image_out, torch.Tensor)
        np.testing.assert_equal(self.image, image_out)

        # test gray array
        image_out = self.ip.to_tensor(self.image[0])
        assert isinstance(image_out, torch.Tensor)
        np.testing.assert_equal(self.image[0], image_out)

        # test RGB tensor
        tensor = torch.from_numpy(self.image)
        image_out = self.ip.to_tensor(tensor)
        assert isinstance(image_out, torch.Tensor)
        np.testing.assert_equal(self.image, image_out)

        # test gray tensor
        tensor = torch.from_numpy(self.image[0])
        image_out = self.ip.to_tensor(tensor)
        assert isinstance(image_out, torch.Tensor)
        np.testing.assert_equal(self.image[0], image_out)

        # test RGB PIL.Image
        pil = Image.fromarray(self.image.transpose((1, 2, 0)))
        image_out = self.ip.to_tensor(pil)
        assert isinstance(image_out, torch.Tensor)
        np.testing.assert_equal(self.image, image_out)

        # test gray PIL.Image
        pil = Image.fromarray(self.image[0])
        image_out = self.ip.to_tensor(pil)
        assert isinstance(image_out, torch.Tensor)
        np.testing.assert_equal(self.image[0], image_out)

    def test_to_pil(self):

        # test RGB array
        image_out = self.ip.to_pil(self.image)
        assert isinstance(image_out, Image.Image)
        np.testing.assert_equal(self.image.transpose((1, 2, 0)), image_out)

        # test gray array
        image_out = self.ip.to_pil(self.image[0])
        assert isinstance(image_out, Image.Image)
        np.testing.assert_equal(self.image[0], image_out)

        # test RGB tensor
        tensor = torch.from_numpy(self.image)
        image_out = self.ip.to_pil(tensor)
        assert isinstance(image_out, Image.Image)
        np.testing.assert_equal(self.image.transpose((1, 2, 0)), image_out)

        # test gray tensor
        tensor = torch.from_numpy(self.image[0])
        image_out = self.ip.to_pil(tensor)
        assert isinstance(image_out, Image.Image)
        np.testing.assert_equal(self.image[0], image_out)

        # test RGB PIL.Image
        pil = Image.fromarray(self.image.transpose((1, 2, 0)))
        image_out = self.ip.to_pil(pil)
        assert isinstance(image_out, Image.Image)
        np.testing.assert_equal(self.image.transpose((1, 2, 0)), image_out)

        # test gray PIL.Image
        pil = Image.fromarray(self.image[0])
        image_out = self.ip.to_pil(pil)
        assert isinstance(image_out, Image.Image)
        np.testing.assert_equal(self.image[0], image_out)

    def test_resize(self):

        size = (10, 12)
        interpolation = 'nearest'

        # test RGB array
        arr1 = self.ip.resize(self.image, size, interpolation)
        arr2 = cv2.resize(self.image.transpose((1, 2, 0)), size[::-1],
                          interpolation=self.ip.str2cv2_interp[interpolation])
        assert isinstance(arr1, np.ndarray)
        np.testing.assert_equal(arr1.transpose((1, 2, 0)), arr2)

        # test gray array
        arr1 = self.ip.resize(self.image[0], size, interpolation)
        arr2 = cv2.resize(self.image[0], size[::-1],
                          interpolation=self.ip.str2cv2_interp[interpolation])
        assert isinstance(arr1, np.ndarray)
        np.testing.assert_equal(arr1, arr2)

        # test RGB tensor
        tensor = torch.from_numpy(self.image)
        tensor1 = self.ip.resize(tensor, size, interpolation)
        tensor2 = cv2.resize(self.image.transpose((1, 2, 0)), size[::-1],
                             interpolation=self.ip.str2cv2_interp[interpolation])
        assert isinstance(tensor1, torch.Tensor)
        np.testing.assert_equal(tensor1, tensor2.transpose((2, 0, 1)))

        # test gray tensor
        tensor = torch.from_numpy(self.image[0])
        tensor1 = self.ip.resize(tensor, size, interpolation)
        tensor2 = cv2.resize(self.image[0], size[::-1],
                             interpolation=self.ip.str2cv2_interp[interpolation])
        assert isinstance(tensor1, torch.Tensor)
        np.testing.assert_equal(tensor1, tensor2)

        # test RGB PIL.Image
        pil = Image.fromarray(self.image.transpose((1, 2, 0)))
        pil1 = self.ip.resize(pil, size, interpolation)
        pil2 = pil.resize(size[::-1], self.ip.str2pil_interp[interpolation])
        assert isinstance(pil1, Image.Image)
        np.testing.assert_equal(pil1, pil2)

        # test gray PIL.Image
        pil = Image.fromarray(self.image[0])
        pil1 = self.ip.resize(pil, size, interpolation)
        pil2 = pil.resize(size[::-1], self.ip.str2pil_interp[interpolation])
        assert isinstance(pil1, Image.Image)
        np.testing.assert_equal(pil1, pil2)

    def test_crop(self):

        box = (0, 0, 2, 4)

        # test RGB array
        arr1 = self.ip.crop(self.image, box)
        arr2 = self.image[:, box[1]:box[3], box[0]:box[2]]
        assert isinstance(arr1, np.ndarray)
        np.testing.assert_equal(arr1, arr2)

        # test gray array
        arr1 = self.ip.crop(self.image[0], box)
        arr2 = self.image[0][box[1]:box[3], box[0]:box[2]]
        assert isinstance(arr1, np.ndarray)
        np.testing.assert_equal(arr1, arr2)

        # test RGB tensor
        tensor = torch.from_numpy(self.image)
        tensor1 = self.ip.crop(tensor, box)
        tensor2 = tensor[:, box[1]:box[3], box[0]:box[2]]
        assert isinstance(tensor1, torch.Tensor)
        torch.equal(tensor1, tensor2)

        # test gray tensor
        tensor = torch.from_numpy(self.image[0])
        tensor1 = self.ip.crop(tensor, box)
        tensor2 = tensor[box[1]:box[3], box[0]:box[2]]
        assert isinstance(tensor1, torch.Tensor)
        torch.equal(tensor1, tensor2)

        # test RGB PIL.Image
        pil = Image.fromarray(self.image.transpose((1, 2, 0)))
        pil1 = self.ip.crop(pil, box)
        pil2 = pil.crop(box)
        assert isinstance(pil1, Image.Image)
        np.testing.assert_equal(pil1, pil2)

        # test gray PIL.Image
        pil = Image.fromarray(self.image[0])
        pil1 = self.ip.crop(pil, box)
        pil2 = pil.crop(box)
        assert isinstance(pil1, Image.Image)
        np.testing.assert_equal(pil1, pil2)

    def test_translate(self):

        #prepare image
        img = np.random.normal(size=(3,50,50))
        bkg = np.zeros((3,224,224))
        
        #prepare parameters
        startpoint = (20,20)
        endpoint = (20,50)
        stride = 2
        
        img_tran = self.ip.translate(img, bkg, startpoint, endpoint, stride)
        
        # assert shape
        num = int((endpoint[1]-startpoint[1])/stride)+1
        assert img_tran.shape[0] == num
        assert (img_tran.shape[2],img_tran.shape[3]) == (bkg.shape[1], bkg.shape[2])
        
        # assert pixel
        pic = img_tran[15]
        plt.imshow(pic.transpose(1,2,0))
        print(pic[:,50,20])
        pixel = pic[0,30,40]
        assert pixel == 0
    
    def test_norm(self):

        # assert L1
        assert self.ip.norm(self.image, 1) == np.abs(self.image).sum()

        # assert L2
        norm1 = self.ip.norm(self.image, 2)
        norm2 = np.sqrt(np.sum([i**2 for i in self.image.ravel()]))
        assert norm1 == norm2

    def test_total_variation(self):

        img = np.array([[1, 2], [3, 4]])
        assert self.ip.total_variation(img) == 6


def test_cross_val_confusion():
    X = np.random.randn(30, 5)
    y = np.random.randint(0, 2, 30)
    svc = SVC()
    cv = 3

    accs_true = cross_val_score(svc, X, y, cv=cv, scoring='accuracy')
    conf_ms, accs_test = db_base.cross_val_confusion(svc, X, y, cv)
    np.testing.assert_equal(accs_test, accs_true)


class TestUnivariateMapping:

    def test_map(self):
        uv = db_base.UnivariateMapping()
        cv = 3
        n_trg = 1
        n_label = 2
        X = np.random.randn(30, 5)
        y_c = np.random.randint(0, n_label, (30, n_trg))
        y_r = np.random.randn(30, n_trg)
        keys = ['score', 'location', 'model', 'conf_m']

        # test corr
        uv.set_estimator('corr')
        uv.set_cv(3)
        map_dict_corr = uv.map(X, y_r)
        assert sorted(keys[:2]) == sorted(map_dict_corr.keys())
        for v in map_dict_corr.values():
            assert v.shape == (n_trg,)

        # test regressor
        uv.set_estimator('glm')
        uv.set_cv(cv)
        uv.set_scoring('correlation')
        map_dict_r = uv.map(X, y_r)
        assert sorted(keys[:3]) == sorted(map_dict_r.keys())
        for k, v in map_dict_r.items():
            if k == 'score':
                assert v.shape == (n_trg, cv)
            else:
                assert v.shape == (n_trg,)

        # test classifier
        uv.set_estimator('lrc')
        uv.set_cv(cv)
        map_dict_c = uv.map(X, y_c)
        assert sorted(keys) == sorted(map_dict_c.keys())
        for k, v in map_dict_c.items():
            if k == 'score':
                assert v.shape == (n_trg, cv)
            elif k == 'conf_m':
                assert v.shape == (n_trg, cv)
                assert v[0][0].shape == (n_label, n_label)
            else:
                assert v.shape == (n_trg,)


class TestMultivariateMapping:

    def test_map(self):
        mv = db_base.MultivariateMapping()
        cv = 3
        n_trg = 2
        n_label = 2
        X = np.random.randn(30, 5)
        Y_c = np.random.randint(0, n_label, (30, n_trg))
        Y_r = np.random.randn(30, n_trg)

        # test classifier
        mv.set_estimator('svc')
        mv.set_cv(cv)
        map_dict_c = mv.map(X, Y_c)
        for trg_idx in range(n_trg):
            conf_ms_true, accs_true = db_base.cross_val_confusion(mv.estimator, X, Y_c[:, trg_idx], cv)
            np.testing.assert_equal(map_dict_c['score'][trg_idx], accs_true)
            for cv_idx in range(cv):
                np.testing.assert_equal(map_dict_c['conf_m'][trg_idx, cv_idx], conf_ms_true[cv_idx])
            coef_test = copy.deepcopy(mv.estimator).fit(X, Y_c[:, trg_idx]).coef_
            np.testing.assert_equal(map_dict_c['model'][trg_idx].coef_, coef_test)

        # test regressor (multi-target flag is True)
        mv.set_estimator('glm')
        mv.set_cv(cv)
        mv.set_scoring('explained_variance')
        map_dict_r1 = mv.map(X, Y_r)
        for trg_idx in range(n_trg):
            scores_true = cross_val_score(mv.estimator, X, Y_r[:, trg_idx], scoring='explained_variance', cv=cv)
            np.testing.assert_almost_equal(map_dict_r1['score'][trg_idx], scores_true, 10)
            coef_test = copy.deepcopy(mv.estimator).fit(X, Y_r[:, trg_idx]).coef_
            np.testing.assert_almost_equal(map_dict_r1['model'][0].coef_[trg_idx], coef_test, 10)

        # test regressor (multi-target flag is False)
        mv.set_estimator(SVR(kernel='linear'))
        mv.set_cv(cv)
        mv.set_scoring('r2')
        map_dict_r2 = mv.map(X, Y_r)
        for trg_idx in range(n_trg):
            scores_true = cross_val_score(mv.estimator, X, Y_r[:, trg_idx], scoring='r2', cv=cv)
            np.testing.assert_almost_equal(map_dict_r2['score'][trg_idx], scores_true, 10)
            coef_test = copy.deepcopy(mv.estimator).fit(X, Y_r[:, trg_idx]).coef_
            np.testing.assert_equal(map_dict_r2['model'][trg_idx].coef_, coef_test)


if __name__ == '__main__':
    pytest.main()
