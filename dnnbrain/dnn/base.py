import os
import cv2
import torch
import numpy as np

from PIL import Image
from os.path import join as pjoin
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import pairwise_distances
from scipy.signal import periodogram
from torchvision import transforms

DNNBRAIN_MODEL = pjoin(os.environ['DNNBRAIN_DATA'], 'models')


def normalize(array):
    """
    Normalize an array's value domain to [0, 1]
    Note: the original normalize function is at dnnbrain/utils/util.py
        but 'from dnnbrain.dnn.core import Mask' in the file causes import conflicts.
        Fix the conflicts in future.

    Parameter:
    ---------
    array[ndarray]: a numpy array

    Return:
    ------
    array[ndarray]: a numpy array after normalization
    """
    array = (array - array.min()) / (array.max() - array.min())

    return array


def array_statistic(arr, method, axis=None, keepdims=False):
    """
    extract statistic of an array

    Parameters:
    ----------
    arr[array]: a numpy array
    method[str]: feature extraction method
    axis[int|tuple]: None or int or tuple of ints
        Axis or axes along which to operate.
        If it's None, operate on the whole array.
    keepdims[bool]: keep the axis which is reduced

    Return:
    ------
    arr[array]: extracted statistic
    """
    if method == 'max':
        arr = np.max(arr, axis, keepdims=keepdims)
    elif method == 'mean':
        arr = np.mean(arr, axis, keepdims=keepdims)
    elif method == 'median':
        arr = np.median(arr, axis, keepdims=keepdims)
    elif method == 'L1':
        arr = np.linalg.norm(arr, 1, axis, keepdims=keepdims)
    elif method == 'L2':
        arr = np.linalg.norm(arr, 2, axis, keepdims=keepdims)
    else:
        raise ValueError('Not supported method:', method)

    return arr


class ImageProcessor:

    def __init__(self):
        self.str2pil_interp = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }

        self.str2cv2_interp = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }

    def _check_image(self, image):
        """
        Check if the image is valid.

        Parameter:
        ---------
        image[ndarray|Tensor|PIL.Image]: image data
            If is ndarray or Tensor, its shape is (height, width) or (3, height, width)
        """
        if isinstance(image, (np.ndarray, torch.Tensor)):
            if image.ndim == 2:
                pass
            elif image.ndim == 3:
                assert image.shape[0] == 3, "RGB channel must be the first axis."
            else:
                raise ValueError("Only two shapes are valid: "
                                 "(height, width) and (3, height, width)")
        elif isinstance(image, Image.Image):
            pass
        else:
            raise TypeError("Only support three types of image: "
                            "ndarray, Tensor and PIL.Image.")

    def to_array(self, image):
        """
        Convert image to array

        Parameter:
        ---------
        image[ndarray|Tensor|PIL.Image]: image data

        Return:
        ------
        arr[ndarray]: image array
        """
        self._check_image(image)

        if isinstance(image, np.ndarray):
            arr = image
        elif isinstance(image, torch.Tensor):
            arr = image.numpy()
        else:
            arr = np.asarray(image)
            if arr.ndim == 3:
                arr = arr.transpose((2, 0, 1))
            elif arr.ndim == 2:
                pass
            else:
                raise ValueError(f"Unsupported number of image dimensions: {arr.ndim}!")

        return arr

    def to_tensor(self, image):
        """
        Convert image to tensor

        Parameter:
        ---------
        image[ndarray|Tensor|PIL.Image]: image data

        Return:
        ------
        tensor[Tensor]: image tensor
        """
        self._check_image(image)

        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = torch.from_numpy(self.to_array(image))

        return tensor

    def to_pil(self, image, normalization=False):
        """
        Convert image to PIL.Image

        Parameter:
        ---------
        image[ndarray|Tensor|PIL.Image]: image data
        normalization[bool]: normalization
            If is True, normalize image data to integers in [0, 255].

        Return:
        ------
        image[PIL.Image]: PIL.Image
        """
        self._check_image(image)

        if normalization:
            image = normalize(self.to_array(image)) * 255
            image = image.astype(np.uint8)

        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = image.transpose((1, 2, 0))
            image = Image.fromarray(image)

        return image

    def resize(self, image, size, interpolation='nearest'):
        """
        Resize image

        Parameters:
        ----------
        image[ndarray|Tensor|PIL.Image]: image data
        size[tuple]: the target size
            as a 2-tuple: (height, width)
        interpolation[str]: interpolation method for resize
            check self.str2pil_interp and self.str2cv2_interp to
            find the available interpolation.

        Return:
        ------
        image[ndarray|Tensor|PIL.Image]: image data after resizing
        """
        self._check_image(image)

        size = size[::-1]
        if isinstance(image, Image.Image):
            image = image.resize(size, self.str2pil_interp[interpolation])
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = cv2.resize(image, size,
                                   interpolation=self.str2cv2_interp[interpolation])
            else:
                image = cv2.resize(image.transpose((1, 2, 0)), size,
                                   interpolation=self.str2cv2_interp[interpolation])
                image = image.transpose((2, 0, 1))
        else:
            image = image.numpy()
            if image.ndim == 2:
                image = cv2.resize(image, size,
                                   interpolation=self.str2cv2_interp[interpolation])
            else:
                image = cv2.resize(image.transpose((1, 2, 0)), size,
                                   interpolation=self.str2cv2_interp[interpolation])
                image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)

        return image

    def crop(self, image, box):
        """
        Crop image with a rectangular region

        Parameters:
        ----------
        image[ndarray|Tensor|PIL.Image]: image data
        box[tuple]: the crop rectangle
            as a (left, upper, right, lower)-tuple

        Return:
        ------
        image[ndarray|Tensor|PIL.Image]: image data after crop
        """
        self._check_image(image)

        if isinstance(image, Image.Image):
            image = image.crop(box)
        else:
            if image.ndim == 2:
                image = image[box[1]:box[3], box[0]:box[2]]
            else:
                image = image[:, box[1]:box[3], box[0]:box[2]]

        return image


class ImageSet:
    """
    Build a dataset to load image
    """
    def __init__(self, img_dir, img_ids, labels=None, transform=None):
        """
        Initialize ImageSet

        Parameters:
        ----------
        img_dir[str]: images' parent directory
        img_ids[list]: Each img_id is a path which can find the image file relative to img_dir.
        labels[list]: Each image's label.
        transform[callable function]: optional transform to be applied on a stimulus.
        """
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.labels = np.ones(len(self.img_ids)) if labels is None else labels
        self.labels = self.labels.astype(np.int64)
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform

    def __len__(self):
        """
        Return the number of images
        """
        return len(self.img_ids)

    def __getitem__(self, indices):
        """
        Get image data and corresponding labels

        Parameter:
        ---------
        indices[int|list|slice]: subscript indices

        Returns:
        -------
        data[tensor]: image data with shape as (n_stim, n_chn, height, weight)
        labels[list]: image labels
        """
        # check availability and do preparation
        if isinstance(indices, int):
            tmp_ids = [self.img_ids[indices]]
            labels = [self.labels[indices]]
        elif isinstance(indices, list):
            tmp_ids = [self.img_ids[idx] for idx in indices]
            labels = [self.labels[idx] for idx in indices]
        elif isinstance(indices, slice):
            tmp_ids = self.img_ids[indices]
            labels = self.labels[indices]
        else:
            raise IndexError("only integer, slices (`:`) and list are valid indices")

        # load data
        data = torch.zeros(0)
        for img_id in tmp_ids:
            image = Image.open(pjoin(self.img_dir, img_id)).convert('RGB')  # load image
            image = self.transform(image)  # transform image
            image = torch.unsqueeze(image, 0)
            data = torch.cat((data, image))

        if data.shape[0] == 1:
            data = data[0]
            labels = labels[0]  # len(labels) == 1

        return data, labels


class VideoSet:
    """
    Dataset for video data
    """
    def __init__(self, vid_file, frame_nums, labels=None, transform=None):
        """
        Parameters:
        ----------
        vid_file[str]: video data file
        frame_nums[list]: sequence numbers of the frames of interest
        labels[list]: each frame's label
        transform[pytorch transform]
        """
        self.vid_cap = cv2.VideoCapture(vid_file)
        self.frame_nums = frame_nums
        self.labels = np.ones(len(self.frame_nums)) if labels is None else labels
        self.labels = self.labels.astype(np.int64)
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform

    def __getitem__(self, indices):
        """
        Get frame data and corresponding labels

        Parameter:
        ---------
        indices[int|list|slice]: subscript indices

        Returns:
        -------
        data[tensor]: frame data with shape as (n_stim, n_chn, height, weight)
        labels[list]: frame labels
        """
        # check availability and do preparation
        if isinstance(indices, int):
            tmp_nums = [self.frame_nums[indices]]
            labels = [self.labels[indices]]
        elif isinstance(indices, list):
            tmp_nums = [self.frame_nums[idx] for idx in indices]
            labels = [self.labels[idx] for idx in indices]
        elif isinstance(indices, slice):
            tmp_nums = self.frame_nums[indices]
            labels = self.labels[indices]
        else:
            raise IndexError("only integer, slices (`:`) and list are valid indices")

        # load data
        data = torch.zeros(0)
        for frame_num in tmp_nums:
            # get frame
            self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
            _, frame = self.vid_cap.read()
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame = self.transform(frame)  # transform frame
            frame = torch.unsqueeze(frame, 0)
            data = torch.cat((data, frame))

        if data.shape[0] == 1:
            data = data[0]
            labels = labels[0]  # len(labels) == 1

        return data, labels

    def __len__(self):
        """
        Return the number of frames
        """
        return len(self.frame_nums)


class UnivariatePredictionModel:

    def __init__(self, model_name=None, cv=3):
        """
        Parameters:
        ----------
        model_name[str]: name of a model used to do prediction
            If is 'corr', it just uses correlation rather than prediction.
        cv[int]: cross validation fold number
        """
        self.set(model_name, cv)

    def set(self, model_name=None, cv=None):
        """
        Set some attributes

        Parameters:
        ----------
        model_name[str]: name of a model used to do prediction
            If is 'corr', it just uses correlation rather than prediction.
        cv[int]: cross validation fold number
        """
        if model_name is None:
            pass
        elif model_name == 'lrc':
            self.model = LogisticRegression()
            self.score_evl = 'accuracy'
        elif model_name == 'svc':
            self.model = SVC(kernel='linear', C=0.025)
            self.score_evl = 'accuracy'
        elif model_name == 'glm':
            self.model = LinearRegression()
            self.score_evl = 'explained_variance'
        elif model_name == 'lasso':
            self.model = Lasso()
            self.score_evl = 'explained_variance'
        elif model_name == 'corr':
            self.model = model_name
            self.score_evl = 'R square'
        else:
            raise ValueError('unsupported model:', model_name)

        if cv is not None:
            self.cv = cv

    def predict(self, X, Y):
        """
        Use all columns of X (one-by-one) to predict each column of Y;
        For each column of Y:
            Find the location of the column of X which has the maximal prediction score;
            Record the location, and corresponding score and model.

        Parameters:
        ----------
        X[ndarray]: shape=(n_sample, n_feature)
        Y[ndarray]: shape=(n_sample, n_target)

        Return:
        ------
        pred_dict[dict]:
            score[ndarray]: shape=(n_target,)
            model[ndarray]: shape=(n_target,)
            location[ndarray]: shape=(n_target,)
            If model_name == 'corr', the score is R square.
                And the model is None.
        """
        assert X.ndim == 2, "X's shape must be (n_sample, n_feature)!"
        assert Y.ndim == 2, "Y's shape must be (n_sample, n_target)!"
        assert X.shape[0] == Y.shape[0], 'X and Y must have the ' \
                                         'same number of samples!'
        n_feat = X.shape[1]
        n_trg = Y.shape[1]
        scores = []
        models = []
        locations = []
        for trg_idx in range(n_trg):
            y = Y[:, trg_idx]
            if self.model == 'corr':
                scores_tmp = pairwise_distances(X.T, y.reshape(1, -1), 'correlation')
                scores_tmp = (1 - scores_tmp.ravel()) ** 2
            else:
                scores_tmp = []
                for feat_idx in range(n_feat):
                    cv_scores = cross_val_score(self.model, X[:, feat_idx][:, None], y,
                                                scoring=self.score_evl, cv=self.cv)
                    scores_tmp.append(np.mean(cv_scores))

            # find maximal score and its location
            max_feat_idx = np.argmax(scores_tmp)
            locations.append(max_feat_idx)
            max_score = scores_tmp[max_feat_idx]
            scores.append(max_score)

            # fit the model with maximal score
            if self.model == 'corr':
                models.append(None)
            else:
                max_model = self.model.fit(X[:, max_feat_idx][:, None], y)
                models.append(deepcopy(max_model))

        pred_dict = {
            'score': np.array(scores),
            'model': np.array(models),
            'location': np.array(locations)
        }
        return pred_dict


class MultivariatePredictionModel:

    def __init__(self, model_name=None, cv=3):
        """
        Parameters:
        ----------
        model_name[str]: name of a model used to do prediction
        cv[int]: cross validation fold number
        """
        self.set(model_name, cv)

    def set(self, model_name=None, cv=None):
        """
        Set some attributes

        Parameters:
        ----------
        model_name[str]: name of a model used to do prediction
        cv[int]: cross validation fold number
        """
        if model_name is None:
            pass
        elif model_name == 'lrc':
            self.model = LogisticRegression()
            self.score_evl = 'accuracy'
        elif model_name == 'svc':
            self.model = SVC(kernel='linear', C=0.025)
            self.score_evl = 'accuracy'
        elif model_name == 'glm':
            self.model = LinearRegression()
            self.score_evl = 'explained_variance'
        elif model_name == 'lasso':
            self.model = Lasso()
            self.score_evl = 'explained_variance'
        else:
            raise ValueError('unsupported model:', model_name)

        if cv is not None:
            self.cv = cv

    def predict(self, X, Y):
        """
        Use all columns of X to predict each column of Y.

        Parameters:
        ----------
        X[ndarray]: shape=(n_sample, n_feature)
        Y[ndarray]: shape=(n_sample, n_target)

        Return:
        ------
        pred_dict[dict]:
            score[ndarray]: shape=(n_target,)
            model[ndarray]: shape=(n_target,)
        """
        assert X.ndim == 2, "X's shape must be (n_sample, n_feature)!"
        assert Y.ndim == 2, "Y's shape must be (n_sample, n_target)!"
        assert X.shape[0] == Y.shape[0], 'X and Y must have the ' \
                                         'same number of samples!'
        n_trg = Y.shape[1]
        scores = []
        models = []
        for trg_idx in range(n_trg):
            y = Y[:, trg_idx]
            cv_scores = cross_val_score(self.model, X, y,
                                        scoring=self.score_evl, cv=self.cv)
            # recording
            scores.append(np.mean(cv_scores))
            models.append(deepcopy(self.model.fit(X, y)))

        pred_dict = {
            'score': np.array(scores),
            'model': np.array(models)
        }
        return pred_dict


def dnn_mask(dnn_acts, channels='all', rows='all', columns='all'):
    """
    Extract DNN activation

    Parameters:
    ----------
    dnn_acts[array]: DNN activation
        A 4D array with its shape as (n_stim, n_chn, n_row, n_col)
    channels[str|list]: channels of interest.
        If is str, it must be 'all' which means all channels.
        If is list, its elements are serial numbers of channels.
    rows[str|list]: rows of interest.
        If is str, it must be 'all' which means all rows.
        If is list, its elements are serial numbers of rows.
    columns[str|list]: columns of interest.
        If is str, it must be 'all' which means all columns.
        If is list, its elements are serial numbers of columns.

    Return:
    ------
    dnn_acts[array]: DNN activation after mask
        a 4D array with its shape as (n_stim, n_chn, n_row, n_col)
    """
    if isinstance(channels, list):
        channels = [chn-1 for chn in channels]
        dnn_acts = dnn_acts[:, channels, :, :]
    if isinstance(rows, list):
        rows = [row-1 for row in rows]
        dnn_acts = dnn_acts[:, :, rows, :]
    if isinstance(columns, list):
        columns = [col-1 for col in columns]
        dnn_acts = dnn_acts[:, :, :, columns]

    return dnn_acts


def dnn_fe(dnn_acts, method, n_feat, axis=None):
    """
    Extract features of DNN activation

    Parameters:
    ----------
    dnn_acts[array]: DNN activation
        a 4D array with its shape as (n_stim, n_chn, n_row, n_col)
    method[str]: feature extraction method, choices=(pca, hist, psd)
        pca: use n_feat principal components as features
        hist: use histogram of activation as features
            Note: n_feat equal-width bins in the given range will be used!
        psd: use power spectral density as features
    n_feat[int]: The number of features to extract
    axis{str}: axis for feature extraction, choices=(chn, row_col)
        If is chn, extract feature along channel axis.
            The result will be an array with shape
            as (n_stim, n_feat, n_row, n_col)
        If is row_col, extract feature alone row and column axis.
            The result will be an array with shape
            as (n_stim, n_chn, n_feat, 1)
        If is None, extract features from the whole layer.
            The result will be an array with shape
            as (n_stim, n_feat, 1, 1)
        We always regard the shape of the result as (n_stim, n_chn, n_row, n_col)

    Return:
    ------
    dnn_acts_new[array]: DNN activation
        a 4D array with its shape as (n_stim, n_chn, n_row, n_col)
    """
    # adjust iterative axis
    n_stim, n_chn, n_row, n_col = dnn_acts.shape
    dnn_acts = dnn_acts.reshape((n_stim, n_chn, n_row*n_col))
    if axis is None:
        dnn_acts = dnn_acts.reshape((n_stim, 1, -1))
    elif axis == 'chn':
        dnn_acts = dnn_acts.transpose((0, 2, 1))
    elif axis == 'row_col':
        pass
    else:
        raise ValueError('not supported axis:', axis)
    _, n_iter, _ = dnn_acts.shape

    # extract features
    dnn_acts_new = np.zeros((n_stim, n_iter, n_feat))
    if method == 'pca':
        pca = PCA(n_components=n_feat)
        for i in range(n_iter):
            dnn_acts_new[:, i, :] = pca.fit_transform(dnn_acts[:, i, :])
    elif method == 'hist':
        for i in range(n_iter):
            for j in range(n_stim):
                dnn_acts_new[j, i, :] = np.histogram(dnn_acts[j, i, :], n_feat)[0]
    elif method == 'psd':
        for i in range(n_iter):
            for j in range(n_stim):
                f, p = periodogram(dnn_acts[j, i, :])
                dnn_acts_new[j, i, :] = p[:n_feat]
    else:
        raise ValueError('not supported method:', method)

    # adjust iterative axis
    if axis is None:
        dnn_acts_new = dnn_acts_new.transpose((0, 2, 1))
        dnn_acts_new = dnn_acts_new[:, :, :, None]
    elif axis == 'chn':
        dnn_acts_new = dnn_acts_new.transpose((0, 2, 1))
        dnn_acts_new = dnn_acts_new.reshape((n_stim, n_feat, n_row, n_col))
    else:
        dnn_acts_new = dnn_acts_new[:, :, :, None]

    return dnn_acts_new
