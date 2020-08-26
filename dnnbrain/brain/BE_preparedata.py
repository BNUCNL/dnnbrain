#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:00:25 2020

@author: gongzhengcenter_xn  zhouming
"""

import numpy as np
import torch
from torch import nn
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class WhatModel(nn.Module):
    """
        Model for feature weighted

    """

    def __init__(self, in_unit, out_unit):
        super().__init__()
        self.in_unit, self.out_unit = in_unit, out_unit
        self.dense = nn.Linear(self.in_unit, self.out_unit)

    def forward(self, X, **kwargs):
        X = X.float()
        X = self.dense(X)

        return X


# get_spatial_kernel(vf_size, kernel_size, center_x, center_y, rf_size) done
def get_spatial_kernel(vf_size, kernel_size, center_x, center_y, rf_size):
    """
    For an image stimuli cover the visual field of **vf_size**(unit of deg.) with
    height=width=**max_resolution**, this method generate the spatial receptive
    field kernel in a gaussian manner with center at (**center_x**, **center_y**),
    and sigma **rf_size**(unit of deg.).

    parameters
    ----------
    vf_size : float

    kernel_size : int
        Usually the origin stimuli resolution.
    center_x : float

    center_y : float

    rf_size : float

    return
    ------
    spatial_kernel : np.ndarray

    """
    # prepare parameter for np.meshgrid
    low_bound = - int(vf_size / 2)
    up_bound = int(vf_size / 2)
    # center at (0,0)
    x = y = np.linspace(low_bound, up_bound, kernel_size)
    y = -y  # adjust orientation
    # generate grid
    xx, yy = np.meshgrid(x, y)
    # prepare for spatial_kernel
    coe = 1 / (2 * np.pi * rf_size ** 2)  # coeficient
    ind = -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * rf_size ** 2)  # gaussian index

    spatial_kernel = coe * np.exp(ind)  # initial spatial_kernel
    # normalize
    spatial_kernel = spatial_kernel / np.sum(spatial_kernel)
    return spatial_kernel


def predict(para_dict, X):
    """
    Predict
    """
    spatial_kernel = get_spatial_kernel(X.shape[-1], para_dict['where_params']['x'],
                                        para_dict['where_params']['y'], para_dict['where_params']['rf_size'])
    X = (X * spatial_kernel).sum(axis=(2, 3))
    X = np.hstack((np.ones(len(X)).reshape(len(X), 1), X))
    weights = para_dict['what_params'].reshape(X.shape[-1], 1)
    prediction = X.dot(weights)
    return prediction


class FeatureWightedModel:
    """
    This method provide training for feature weighted receptive field model.

    """

    def __init__(self, lr=0.1, optimizer=torch.optim.SGD, epochs=20, penalty=0,
                 spatial_search=True):
        self.lr = lr
        self.optimizer = optimizer
        self.epochs = epochs
        self.penalty = penalty
        self.spatial_search = spatial_search
        self.spatial_x, self.spatial_y, self.spatial_size = None, None, None
        if self.spatial_search:
            self.set_grid(np.linspace(-9, 9, 10), np.linspace(-9, 9, 10), 2 ** np.linspace(-0.5145, 3, 8))

        self.what_model = None
        self.model = None
        self.history = []
        self.max_score = None
        self.best_param_dict = {}

    def set_grid(self, x, y, size):
        """
        grid search for spatial kernel.
        default is

        """
        self.spatial_x = x
        self.spatial_y = y
        self.spatial_size = size

    def make_spatial_kernel(self, map_size, center_x, center_y, rf_size):
        """
        This method is to generate the spatial receptive field kernel.

        parameters
        ----------
        map_size : int
            Size of feature.
        center_x : float

        center_y : float

        rf_size : float

        """
        # prepare parameter for meshgrid
        low_bound = - int(map_size / 2)
        up_bound = int(map_size / 2)
        # center at (0,0)
        x = y = np.linspace(low_bound, up_bound, map_size)
        y = -y  # adjust orientation
        # generate grid
        xx, yy = np.meshgrid(x, y)
        # prepare for spatial_kernel
        coe = 1 / (2 * np.pi * rf_size ** 2)  # coeficient
        ind = -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * rf_size ** 2)  # gaussian index
        spatial_kernel = coe * np.exp(ind)  # initial spatial_kernel
        # normalize
        spatial_kernel = spatial_kernel / np.sum(spatial_kernel)
        return spatial_kernel

    def make_feature_vector(self, maps, center_x, center_y, rf_size):
        """

        parameters
        ----------
        maps : tensor
            With shape as (n_samples, n_feature, height, width).
        center_x : float

        center_y : float

        rf_size : float

        return
        -------
        feature_vectors : array

        """
        # initialize feature vector
        feature_vectors = np.array([])
        # if feature maps are in list
        if type(maps) == list:
            for i in range(len(maps)):
                f_map = maps[i]
                map_size = f_map.shape[-1]
                # make the specific kernel
                kernel = self.make_spatial_kernel(map_size, center_x, center_y, rf_size)
                feature_vector = np.sum(f_map * kernel, axis=(2, 3))
                feature_vectors = np.hstack((feature_vectors, feature_vector))
        elif type(maps) == np.ndarray:
            # input array height
            map_size = maps.shape[-1]
            kernel = self.make_spatial_kernel(map_size, center_x, center_y, rf_size)
            feature_vectors = np.sum(maps * kernel, axis=(2, 3))

        return feature_vectors

    def make_model(self, in_dim, out_dim, lr,
                   optimizer=torch.optim.SGD, epochs=20, penalty=0):
        """
        Make a neuron network model

        parameters
        ----------
        in_dim : int
            Number of features.
        out_dim : int
            Number of voxels.
        lr : float
            Learning rate for optimizer.
        optimizer : class
            Torch optimizer.
        epochs : int
            Epochs for training model.
        penalty : float
            L2 regularization lambda.
        """

        self.what_model = WhatModel(in_unit=in_dim, out_unit=out_dim)
        # skorch make a regression
        self.model = NeuralNetRegressor(module=self.what_model, max_epochs=epochs,
                                        lr=lr, optimizer=optimizer,
                                        optimizer__weight_decay=penalty, verbose=0)

    def fit_model(self, feature_vectors, y, center_x, center_y, rf_size):
        """


        feature_vectors : array

        y : array

        """

        self.model.fit(feature_vectors, y)
        self.history.append({
            'kernel': [center_x, center_y, rf_size],
            'weights': np.array(self.what_model.dense.weight.detach()),
            'bias': np.array(self.what_model.dense.bias.detach()),
            'valid_score': self.model.history_[-1]['valid_loss']
        })

    def sort_history(self):
        """
        sort history list to get the best parameters & weights

        """
        valid_scores = []
        for i in range(len(self.history)):
            cur_dict = self.history[i]
            valid_scores.append(cur_dict['valid_score'])
        # because 'valid_score' == valid_loss
        max_position = int(np.argmin(np.array(valid_scores)))
        self.max_score = valid_scores[max_position]
        x, y, rf_size = self.history[max_position]['kernel']
        bias = self.history[max_position]['bias']
        weights = self.history[max_position]['weights']
        what_params = np.hstack((bias.reshape(-1), weights.reshape(-1)))
        self.best_param_dict = {'where_params': {'x': x, 'y': y, 'rf_size': rf_size},
                                'what_params': what_params}

    def fit(self, X, y):
        """

        parameters
        -----------
        X : ndarray, list
            Features shape as (n_sample, n_feat, width, height),
            only 2d is supported.
        y : ndarray
            Voxel value shape as (n_sample, n_voxel).

        """
        # get parameters for nn structure
        n_feat, n_voxel = 0, 0
        if type(X) == list:
            num_feat = []
            for i in range(X):
                num_feat.append(int(X[i].shape[1]))
            n_feat = int(np.sum(num_feat))
            n_voxel = y.shape[1]
        elif type(X) == np.ndarray:
            n_feat, n_voxel = X.shape[1], y.shape[1]
        # construct the what model
        self.make_model(in_dim=n_feat, out_dim=n_voxel, lr=self.lr,
                        optimizer=self.optimizer, epochs=self.epochs, penalty=self.penalty)
        # process & fit data
        if self.spatial_search:
            i = 0
            for s_x in self.spatial_x:
                for s_y in self.spatial_y:
                    for s_rf_size in self.spatial_size:
                        feature_vectors = self.make_feature_vector(X, s_x, s_y, s_rf_size)
                        self.fit_model(feature_vectors, y, s_x, s_y, s_rf_size)
                        i += 1
                        # verbose
                        if i % 100 == 0:
                            print('### Model', i)
        # find best one
        self.sort_history()


class SpatialConvolve(BaseEstimator, TransformerMixin):
    """
    The method is to process feature map data.
    This is a transformer based on sk-learn base.
    """
    def __init__(self, center_x, center_y, size):
        self.center_x = center_x
        self.center_y = center_y
        self.size = size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None,):
        """

        """
        # initialize
        feature_vectors = np.array([])
        if not ((type(X) == list) ^ (type(X) == np.ndarray)):
            raise AssertionError('Data type of X is not supported, '
                                 'Please check only list or numpy.ndarray')
        elif type(X) == list:
            # generate vectors
            for i in range(len(X)):
                f_map = X[i]
                map_size = f_map.shape[-1]
                # make the specific kernel
                kernel = get_spatial_kernel(20, map_size, self.center_x, self.center_y, self.size)
                feature_vector = np.sum(f_map * kernel, axis=(2, 3))
                feature_vectors = np.hstack((feature_vectors, feature_vector))
        elif type(X) == np.ndarray:
            # initialize
            feature_vectors = np.array([])
            # input array height
            map_size = X.shape[-1]
            kernel = get_spatial_kernel(20, map_size, self.center_x, self.center_y, self.size)
            feature_vectors = np.sum(X * kernel, axis=(2, 3))

        return feature_vectors


class FeatureWightedModelV2:
    """
    Second edition of fwpRF model
    This method provide training for feature weighted receptive field model.

    """

    def __init__(self, lr=0.1, optimizer=torch.optim.SGD, epochs=20, penalty=0,
                 spatial_search=True):
        self.lr = lr
        self.optimizer = optimizer
        self.epochs = epochs
        self.penalty = penalty
        self.spatial_search = spatial_search
        self.spatial_x, self.spatial_y, self.spatial_size = None, None, None
        if self.spatial_search:
            self.set_grid(np.linspace(-9, 9, 10), np.linspace(-9, 9, 10), 2 ** np.linspace(-0.5145, 3, 8))

        self.what_model = None
        self.model = None
        self.history = []
        self.max_score = None
        self.best_param_dict = {}

    def set_grid(self, x, y, size):
        """
        grid search for spatial kernel.
        default is

        """
        self.spatial_x = x
        self.spatial_y = y
        self.spatial_size = size

    def make_model(self, in_dim, out_dim, lr,
                   optimizer=torch.optim.SGD, epochs=20, penalty=0):
        """
        Make a neuron network model

        parameters
        ----------
        in_dim : int
            Number of features.
        out_dim : int
            Number of voxels.
        lr : float
            Learning rate for optimizer.
        optimizer : class
            Torch optimizer.
        epochs : int
            Epochs for training model.
        penalty : float
            L2 regularization lambda.
        """

        self.what_model = WhatModel(in_unit=in_dim, out_unit=out_dim)
        # skorch make a regression
        self.model = NeuralNetRegressor(module=self.what_model, max_epochs=epochs,
                                        lr=lr, optimizer=optimizer,
                                        optimizer__weight_decay=penalty, verbose=0)

    def fit_model(self, feature_vectors, y, center_x, center_y, rf_size):
        """


        feature_vectors : array

        y : array

        """

        self.model.fit(feature_vectors, y)
        self.history.append({
            'kernel': [center_x, center_y, rf_size],
            'weights': np.array(self.what_model.dense.weight.detach()),
            'bias': np.array(self.what_model.dense.bias.detach()),
            'valid_score': self.model.history_[-1]['valid_loss']
        })

    def sort_history(self):
        """
        sort history list to get the best parameters & weights

        """
        valid_scores = []
        for i in range(len(self.history)):
            cur_dict = self.history[i]
            valid_scores.append(cur_dict['valid_score'])
        # because 'valid_score' == valid_loss
        max_position = int(np.argmin(np.array(valid_scores)))
        self.max_score = valid_scores[max_position]
        x, y, rf_size = self.history[max_position]['kernel']
        bias = self.history[max_position]['bias']
        weights = self.history[max_position]['weights']
        what_params = np.hstack((bias.reshape(-1), weights.reshape(-1)))
        self.best_param_dict = {'where_params': {'x': x, 'y': y, 'rf_size': rf_size},
                                'what_params': what_params}

    def fit(self, X, y):
        """

        parameters
        -----------
        X : ndarray, list
            Features shape as (n_sample, n_feat, width, height),
            only 2d is supported.
        y : ndarray
            Voxel value shape as (n_sample, n_voxel).

        """
        # get parameters for nn structure
        n_feat, n_voxel = 0, 0
        if type(X) == list:
            num_feat = []
            for i in range(X):
                num_feat.append(int(X[i].shape[1]))
            n_feat = int(np.sum(num_feat))
            n_voxel = y.shape[1]
        elif type(X) == np.ndarray:
            n_feat, n_voxel = X.shape[1], y.shape[1]
        # construct the what model
        self.make_model(in_dim=n_feat, out_dim=n_voxel, lr=self.lr,
                        optimizer=self.optimizer, epochs=self.epochs, penalty=self.penalty)
        # process & fit data
        if self.spatial_search:
            i = 0
            for s_x in self.spatial_x:
                for s_y in self.spatial_y:
                    for s_rf_size in self.spatial_size:
                        feature_vectors = self.make_feature_vector(X, s_x, s_y, s_rf_size)
                        self.fit_model(feature_vectors, y, s_x, s_y, s_rf_size)
                        i += 1
                        # verbose
                        if i % 100 == 0:
                            print('### Model', i)
        # find best one
        self.sort_history()


# transfer_axis(vf_axis, vf_size, tar_size) done
def transfer_axis(vf_axis, vf_size, tar_size):
    """
    Transfer visual field location to pixel space location.
    In the range of visual field, given as **vf_size**,
    for an specific **vf_axis**(unit of deg.), this method help
    to compute the discrete locations correspond to feature map with
    height = width = **tar_size**.
    The visual field axis should be smaller than range of vf_size.

    vf_axis : array

    vf_size : float

    tar_size : int

    return
    ------
    index
    """
    # check data requirement
    if np.max(np.abs(vf_axis)) > vf_size:
        raise AssertionError('Check your vf_axis, ')
    else:
        # initialize **index**
        index = []
        # norm axis to [0, 1]
        norm_vf_axis = (vf_axis + vf_size/2)/vf_size
        # compute corresponding location within resolution
        tar = (tar_size-1) * norm_vf_axis
        # floor the **tar** to get discrete **index**
        if type(tar) == np.ndarray:
            index = np.floor(tar)
        elif type(tar) == float:
            index = int(tar)
        return index


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # parameter setting
    num_sample = 1200
    num_feature = 20
    resolution = 10
    # generate random data #
    np.random.seed(1)
    fw = np.random.randn(num_feature).astype(np.float32)
    X_all = np.random.rand(num_sample, num_feature, resolution, resolution).astype(np.float32)
    # voxel 1 parameters
    where_para1 = get_spatial_kernel(vf_size=20, kernel_size=resolution, center_x=-3, center_y=-5, rf_size=3.5).astype(np.float32)
    what_para1 = (4 * fw * (np.abs(4 * fw) < 4)).reshape(num_feature, 1)
    vox1 = ((X_all * where_para1).sum(axis=(2, 3))).dot(what_para1) + np.random.randn(num_sample, 1)
    print(vox1.shape)
    # voxel 2 parameters
    where_para2 = get_spatial_kernel(vf_size=20, kernel_size=resolution, center_x=-3, center_y=5, rf_size=3.6).astype(np.float32)
    what_para2 = ((4 * fw + 1) * (np.abs(4 * fw) < 2.8)).reshape(num_feature, 1)
    vox2 = ((X_all * where_para2).sum(axis=(2, 3))).dot(what_para2) + np.random.randn(num_sample, 1)
    print(vox2.shape)
    # concatenate vox1 & vox2
    y = np.hstack((vox1, vox2)).astype(np.float32)
    print(y.shape)

    X_train = X_all[:int(num_sample*0.85), :]
    y_train = y[:int(num_sample*0.85), :]
    X_test = X_all[int(num_sample*0.85):, :]
    y_test = y[int(num_sample*0.85):, :]

    # # train single model one-by-one
    # # vox1
    # fw_model_sing_vo1 = FeatureWightedModel(lr=0.1, optimizer=torch.optim.SGD, epochs=100,
    #                                         penalty=0, spatial_search=True)
    # fw_model_sing_vo1.fit(X_train, y_train[:, 0].reshape(len(y_train), 1))
    # print(fw_model_sing_vo1.best_param_dict['where_params'])
    # # vox2
    # fw_model_sing_vo2 = FeatureWightedModel(lr=0.1, optimizer=torch.optim.SGD, epochs=10,
    #                                         penalty=1, spatial_search=True)
    # fw_model_sing_vo2.fit(X_train, y_train[:, 1].reshape(len(y_train), 1))
    # print(fw_model_sing_vo2.best_param_dict)
    # # train model parallel BE_preparedata(1)
    # fw_model_para = FeatureWightedModel(lr=0.1, optimizer=torch.optim.SGD, epochs=100,
    #                                     penalty=1, spatial_search=True)
    # fw_model_para.fit(X_train, y_train)
    # fw_model_para.fit(X_train, y_train)
    # print(fw_model_para.best_param_dict['where_params'])

    # ########################################################
    lr, optimizer, epochs, penalty = 0.1, torch.optim.SGD, 20, 0
    what_net = WhatModel(in_unit=20, out_unit=1)

    fw_what_model = NeuralNetRegressor(module=what_net, max_epochs=epochs, lr=lr, optimizer=optimizer,
                                       optimizer__weight_decay=penalty, verbose=0)
    fw_pipe = Pipeline([
        ('where', SpatialConvolve(-9, -9, 0.7)),
        ('what', fw_what_model)
    ])

    parameters = {
        'where__center_x': np.linspace(-9, 9, 2),
        'where__center_y': np.linspace(-9, 9, 2),
        'where__size': 2**np.linspace(-0.5, 3, 2),
        'what__lr': [0.01, 0.1],
        'what__optimizer__weight_decay': [0, 0.1]
    }

    gs = GridSearchCV(fw_pipe, parameters, verbose=2, refit=True, cv=2)
    gs.fit(X_train, y_train[:, 0].reshape(len(y_train), 1))


