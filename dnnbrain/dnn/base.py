import os
import cv2
import torch
import numpy as np

from PIL import Image
from os.path import join as pjoin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy.signal import periodogram
from torchvision import transforms
from torchvision import models as torch_models
from dnnbrain.dnn import models as db_models

DNNBRAIN_MODEL = pjoin(os.environ['DNNBRAIN_DATA'], 'models')


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
            image = Image.open(pjoin(self.img_dir, img_id))  # load image
            image = self.transform(image)  # transform image
            image = torch.unsqueeze(image, 0)
            data = torch.cat((data, image))

        if data.shape[0] == 1:
            data = data[0]

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

        return data, labels

    def __len__(self):
        """
        Return the number of frames
        """
        return len(self.frame_nums)


class DNNLoader:
    """
    Load DNN model and initiate some information
    """

    def __init__(self, net):
        """
        Load neural network model by net name

        Parameter:
        ---------
        net[str]: a neural network's name
        """
        if net == 'alexnet':
            self.model = torch_models.alexnet()
            self.model.load_state_dict(torch.load(
                pjoin(DNNBRAIN_MODEL, 'alexnet_param.pth')))
            self.layer2loc = {'conv1': ('features', '0'), 'conv1_relu': ('features', '1'),
                              'conv1_maxpool': ('features', '2'), 'conv2': ('features', '3'),
                              'conv2_relu': ('features', '4'), 'conv2_maxpool': ('features', '5'),
                              'conv3': ('features', '6'), 'conv3_relu': ('features', '7'),
                              'conv4': ('features', '8'), 'conv4_relu': ('features', '9'),
                              'conv5': ('features', '10'), 'conv5_relu': ('features', '11'),
                              'conv5_maxpool': ('features', '12'), 'fc1': ('classifier', '1'),
                              'fc1_relu': ('classifier', '2'), 'fc2': ('classifier', '4'),
                              'fc2_relu': ('classifier', '5'), 'fc3': ('classifier', '6')}
            self.img_size = (224, 224)
        elif net == 'vgg11':
            self.model = torch_models.vgg11()
            self.model.load_state_dict(torch.load(
                pjoin(DNNBRAIN_MODEL, 'vgg11_param.pth')))
            self.layer2loc = None
            self.img_size = (224, 224)
        elif net == 'vggface':
            self.model = db_models.Vgg_face()
            self.model.load_state_dict(torch.load(
                pjoin(DNNBRAIN_MODEL, 'vgg_face_dag.pth')))
            self.layer2loc = None
            self.img_size = (224, 224)
        else:
            raise ValueError("Not supported net name:", net)


class Classifier:
    """
    Encapsulate some classifier models of scikit-learn
    """

    def __init__(self, name=None):
        """
        Parameter:
        ---------
        name[str]: the name of a classifier model
        """
        self.model = None
        self.score_evl = 'accuracy'
        if name is not None:
            self.set(name)

    def set(self, name):
        """
        Set model

        Parameter:
        ---------
        name[str]: the name of a classifier model
        """
        if name == 'lrc':
            self.model = LogisticRegression()
        elif name == 'svc':
            self.model = SVC(kernel='linear', C=0.025)
        else:
            raise ValueError('unsupported model:', name)

    def fit(self, X, y):
        """
        Fit model

        Parameters:
        ----------
        X[array]: data with shape as (n_samples, n_features)
        y[array]: target values with shape as (n_samples,)

        Return:
        ------
        self[Classifier]: an instance of self
        """
        self.model.fit(X, y)

        return self

    def predict(self, X):
        """
        Do prediction

        Parameter:
        ---------
        X[array]: the data with shape as (n_samples, n_features)

        Return:
        ------
            [array]: predicted values with shape as (n_samples,)
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Evaluate the model

        Parameters:
        ----------
        X[array]: data with shape as (n_samples, n_features)
        y[array]: true values with shape as (n_samples,)

        Return:
        ------
            [float]: Mean accuracy of self.predict(X) wrt. y.
        """
        return self.model.score(X, y)

    def cross_val_score(self, X, y, cv=3):
        """
        Evaluate the model through cross validation.

        Parameters:
        ----------
        X[array]: data with shape as (n_samples, n_features)
        y[array]: true values with shape as (n_samples,)
        cv[int]: the number of folds

        Return:
        ------
        scores[array]: array of float with shape as (cv,)
        """
        scores = cross_val_score(self.model, X, y,
                                 scoring=self.score_evl, cv=cv)
        return scores


class Regressor:
    """
    Encapsulate some regressor models of scikit-learn
    """

    def __init__(self, name=None):
        """
        Parameter:
        ---------
        name[str]: the name of a classifier model
        """
        self.model = None
        self.score_evl = 'explained_variance'
        if name is not None:
            self.set(name)

    def set(self, name):
        """
        Set model

        Parameter:
        ---------
        name[str]: the name of a regreesor model
        """
        if name == 'glm':
            self.model = LinearRegression()
        elif name == 'lasso':
            self.model = Lasso()
        else:
            raise ValueError('unsupported model:', name)

    def fit(self, X, y):
        """
        Fit model

        Parameters:
        ----------
        X[array]: data with shape as (n_samples, n_features)
        y[array]: target values with shape as (n_samples,)

        Return:
        ------
        self[Classifier]: an instance of self
        """
        self.model.fit(X, y)

        return self

    def predict(self, X):
        """
        Do prediction

        Parameter:
        ---------
        X[array]: data with shape as (n_samples, n_features)

        Return:
        ------
            [array]: predicted values with shape as (n_samples,)
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Evaluate the model

        Parameters:
        ----------
        X[array]: data with shape as (n_samples, n_features)
        y[array]: true values with shape as (n_samples,)

        Return:
        ------
            [float]: R^2 of self.predict(X) wrt. y.
        """
        return self.model.score(X, y)

    def cross_val_score(self, X, y, cv=3):
        """
        Evaluate the model through cross validation.

        Parameters:
        ----------
        X[array]: data with shape as (n_samples, n_features)
        y[array]: true values with shape as (n_samples,)
        cv[int]: the number of folds

        Return:
        ------
        scores[array]: array of float with shape as (cv,)
        """
        scores = cross_val_score(self.model, X, y,
                                 scoring=self.score_evl, cv=cv)
        return scores


def dnn_mask(dnn_acts, channels=None, rows=None, columns=None):
    """
    Extract DNN activation

    Parameters:
    ----------
    dnn_acts[array]: DNN activation
        A 4D array with its shape as (n_stim, n_chn, n_row, n_col)
    channels[list]: sequence numbers of channels of interest.
    rows[list]: sequence numbers of rows of interest.
    columns[list]: sequence numbers of columns of interest.

    Return:
    ------
    dnn_acts[array]: DNN activation after mask
        a 4D array with its shape as (n_stim, n_chn, n_row, n_col)
    """
    if channels is not None:
        channels = [chn-1 for chn in channels]
        dnn_acts = dnn_acts[:, channels, :, :]
    if rows is not None:
        rows = [row-1 for row in rows]
        dnn_acts = dnn_acts[:, :, rows, :]
    if columns is not None:
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
