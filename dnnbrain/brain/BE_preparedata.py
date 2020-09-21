#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on

@author: gongzhengxin  zhouming
"""

import numpy as np
import torch
from torch import nn
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# fully connected layer
class FeatureProcessor(nn.Module):
    """
        FeatureProcessor is the fully connected layer of feature processing.
        Model for feature weighted

    """

    def __init__(self, in_unit, out_unit):
        super().__init__()
        self.in_unit, self.out_unit = in_unit, out_unit
        self.dense = nn.Linear(self.in_unit, self.out_unit)

    def forward(self, features, **kwargs):
        features = features.float()
        response = self.dense(features)

        return response


# Receptive field model
class ReceptiveFieldProcessor(BaseEstimator, TransformerMixin):
    """
    The method is to process feature map data.
    This is a transformer based on sk-learn base.

    parameter
    ---------
    center_x : float / array
    center_y : float / array
    size : float / array

    return
    ------
    feature_vector :
    """

    def __init__(self, vf_size, center_x, center_y, rf_size):
        self.vf_size = vf_size
        self.center_x = center_x
        self.center_y = center_y
        self.rf_size = rf_size

    def get_spatial_kernel(self, kernel_size):
        """
        For an image stimuli cover the visual field of **vf_size**(unit of deg.) with
        height=width=**max_resolution**, this method generate the spatial receptive
        field kernel in a gaussian manner with center at (**center_x**, **center_y**),
        and sigma **rf_size**(unit of deg.).

        parameters
        ----------
        kernel_size : int
            Usually the origin stimuli resolution.

        return
        ------
        spatial_kernel : np.ndarray

        """
        # prepare parameter for np.meshgrid
        low_bound = - int(self.vf_size / 2)
        up_bound = int(self.vf_size / 2)
        # center at (0,0)
        x = y = np.linspace(low_bound, up_bound, kernel_size)
        y = -y  # adjust orientation
        # generate grid
        xx, yy = np.meshgrid(x, y)
        # prepare for spatial_kernel
        coe = 1 / (2 * np.pi * self.rf_size ** 2)  # coeficient
        ind = -((xx - self.center_x) ** 2 + (yy - self.center_y) ** 2) / (2 * self.rf_size ** 2)  # gaussian index

        spatial_kernel = coe * np.exp(ind)  # initial spatial_kernel
        # normalize
        spatial_kernel = spatial_kernel / np.sum(spatial_kernel)
        return spatial_kernel

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, ):
        """

        """
        # initialize
        feature_vectors = np.array([])
        if not ((type(X) == list) ^ (type(X) == np.ndarray)):
            raise AssertionError('Data type of X is not supported, '
                                 'Please check only list or numpy.ndarray')
        elif type(X) == list:
            feature_vectors = np.zeors((len(X[0]), 1))
            # to generate vectors
            for i in range(len(X)):
                f_map = X[i]
                map_size = f_map.shape[-1]
                # make the specific kernel
                kernel = self.get_spatial_kernel(map_size)
                feature_vector = np.sum(f_map * kernel, axis=(2, 3))
                feature_vectors = np.hstack((feature_vectors, feature_vector))
            feature_vectors = np.delete(feature_vectors, 0, axis=1)
        elif type(X) == np.ndarray:
            # input array height
            map_size = X.shape[-1]
            kernel = self.get_spatial_kernel(map_size)
            feature_vectors = np.sum(X * kernel, axis=(2, 3))

        return feature_vectors


# Feature weighted model
class FeatureWightedModel:
    """
    Second edition of fwpRF model
    This method provide training for feature weighted receptive field model.

    """

    def __init__(self, lr=0.1, optimizer=torch.optim.SGD, epochs=20, penalty=0,
                 receptive_field=True, vf_size=None):

        #
        self.lr = lr
        self.optimizer = optimizer
        self.epochs = epochs
        self.penalty = penalty
        self.receptive_field = receptive_field
        #
        self.param_dict = {}
        if self.receptive_field:
            self.vf_size = vf_size
            self._set_search_grid(self._to_list(lr), self._to_list(optimizer), self._to_list(epochs),
                                  self._to_list(penalty), spatial_x=np.linspace(-9, 9, 10),
                                  spatial_y=np.linspace(-9, 9, 10), spatial_size=2 ** np.linspace(-0.5145, 3, 8))
        else:
            self._set_search_grid(self._to_list(lr), self._to_list(optimizer),
                                  self._to_list(epochs), self._to_list(penalty))

        self.grid_search = None
        self.feature_processor, self.what_model = None, None
        self.all_vox_best_params = []

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
        n_feat, n_vox = 0, 0
        if type(X) == list:
            num_feat = []
            for i in range(X):
                num_feat.append(int(X[i].shape[1]))
            n_feat = int(np.sum(num_feat))
            n_vox = y.shape[1]
        elif type(X) == np.ndarray:
            n_feat, n_vox = X.shape[1], y.shape[1]
        # construct the what model
        if n_vox > 1:
            self._make_model(in_dim=n_feat, out_dim=1, lr=self.lr,
                            optimizer=self.optimizer, epochs=self.epochs, penalty=self.penalty)
            print('=================train end======================')
            for i in range(n_vox):
                print('voxel', i + 1, 'now')
                vox_y = y[:, 0].reshape((len(y), 1))
                self.grid_search.fit(X, vox_y)
                params = self.grid_search.best_params_
                center_x, center_y = params['where__center_x'], params['where__center_y']
                rf_size = params['where__rf_size']

                bias = np.array(self.feature_processor.dense.bias.detach())
                weights = np.array(self.feature_processor.dense.weight.detach())
                what_params = np.hstack((bias.reshape(-1), weights.reshape(-1)))
                best_param_dict = {
                    'vox_idx': i + 1,
                    'kernel': [center_x, center_y, rf_size],
                    'weights': weights,
                    'bias': bias,
                    'valid_score': self.grid_search.best_score_,
                    'where_params': {'x': center_x, 'y': center_y, 'rf_size': rf_size},
                    'what_params': what_params
                }
                self.all_vox_best_params.append(best_param_dict)
            print('=================train end======================')

        else:
            self._make_model(in_dim=n_feat, out_dim=n_vox, lr=self.lr,
                            optimizer=self.optimizer, epochs=self.epochs, penalty=self.penalty)
            # process & fit data
            print('=================train start======================')
            self.grid_search.fit(X, y)
            params = self.grid_search.best_params_
            center_x, center_y = params['where__center_x'], params['where__center_y']
            rf_size = params['where__size']
            best_param_dict = {
                'vox_idx': 'Only this one',
                'kernel': [center_x, center_y, rf_size],
                'weights': np.array(self.what_model.dense.weight.detach()),
                'bias': np.array(self.what_model.dense.bias.detach()),
                'valid_score': self.grid_search.best_score_
            }
            self.all_vox_best_params.append(best_param_dict)
            print('=================train end======================')

    def predict(self, X):
        """
        Predict
        """
        predictions = np.zeros((len(X), len(self.all_vox_best_params)))
        for i in range(len(self.all_vox_best_params)):
            para_dict = self.all_vox_best_params[i]
            center_x, center_y = para_dict['where_params']['x'], para_dict['where_params']['y']
            rf_size = para_dict['where_params']['rf_size']
            rf_processor = ReceptiveFieldProcessor(vf_size=self.vf_size, center_x=center_x, center_y=center_y,
                                                   rf_size=rf_size)
            vectors = rf_processor.transform(X)
            vectors = np.hstack((np.ones(len(vectors)).reshape(len(vectors), 1), vectors))
            weights = para_dict['what_params'].reshape(vectors.shape[-1], 1)
            prediction = vectors.dot(weights).reshape(-1)
            predictions[:, i] = prediction

        return predictions

    def set_grid_params(self, **params):
        target_keys = []
        for key_name in list(self.param_dict.keys()):
            target_keys.append(key_name.partition('__')[-1])
        for key, value in params.items():
            if key in target_keys:
                for i in range(len(target_keys)):
                    if target_keys[i] == key:
                        idx = i
                        self.param_dict[list(self.param_dict.keys())[i]] = value
                        break

    def _to_list(self, param):
        """
        For param_dict.
        """
        if not (type(param) == list) ^ (type(param) == np.ndarray):
            param = [param]
            return param
        else:
            return param

    def _set_search_grid(self, lr, optimizer, epochs, penalty, **spatial_params):
        """
        Construct grid search parameters dict.

        """
        if bool(spatial_params):
            if all([spatial_params, 'spatial_x' in spatial_params.keys(),
                    'spatial_y' in spatial_params.keys(), 'spatial_size' in spatial_params.keys()]):
                self.param_dict = {
                    'where__center_x': spatial_params['spatial_x'],
                    'where__center_y': spatial_params['spatial_y'],
                    'where__rf_size': spatial_params['spatial_size'],

                    'what__lr': lr,
                    'what__optimizer': optimizer,
                    # 'what__epochs': epochs,
                    'what__optimizer__weight_decay': penalty
                }
            else:
                raise AssertionError('Please check your spatial_params dict, keys\'spatial_x\''
                                     '\'spatial_y\',\'spatial_size\',all should be included.')
        elif not bool(spatial_params):
            self.param_dict = {
                'what__lr': lr,
                'what__optimizer': optimizer,
                # 'what__epochs': epochs,
                'what__optimizer__weight_decay': penalty
            }

    def _make_model(self, in_dim, out_dim, lr,
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
        if self.receptive_field:
            self.feature_processor = FeatureProcessor(in_unit=in_dim, out_unit=out_dim)
            # skorch make a regression
            self.what_model = NeuralNetRegressor(module=self.feature_processor, max_epochs=epochs,
                                                 lr=lr, optimizer=optimizer,
                                                 optimizer__weight_decay=penalty, verbose=0)
            receptive_field = ReceptiveFieldProcessor(vf_size=self.vf_size, center_x=0, center_y=0, rf_size=0)
            fw_model = Pipeline([
                ('where', receptive_field),
                ('what', self.what_model)
            ])
            self.grid_search = GridSearchCV(fw_model, self.param_dict, verbose=2, refit=True, cv=2)
        else:
            self.feature_processor = FeatureProcessor(in_unit=in_dim, out_unit=out_dim)
            # skorch make a regression
            self.what_model = NeuralNetRegressor(module=self.feature_processor, max_epochs=epochs,
                                                 lr=lr, optimizer=optimizer,
                                                 optimizer__weight_decay=penalty, verbose=0)
            fw_model = Pipeline([
                ('what', self.what_model)
            ])
            self.grid_search = GridSearchCV(fw_model, self.param_dict, verbose=2, refit=True, cv=2)


if __name__ == '__main__':
    # parameter setting
    num_sample = 1200
    num_feature = 1
    resolution = 10
    # generate random data #
    np.random.seed(1)
    fw = np.random.randn(num_feature).astype(np.float32)
    X_all = np.random.rand(num_sample, num_feature, resolution, resolution).astype(np.float32)
    # voxel 1 parameters
    where_para1 = ReceptiveFieldProcessor(vf_size=20, center_x=-3,
                                          center_y=-5, rf_size=3.5).get_spatial_kernel(kernel_size=resolution)
    where_para1 = where_para1.astype(np.float32)
    what_para1 = (4 * fw * (np.abs(4 * fw) < 8)).reshape(num_feature, 1)
    vox1 = ((X_all * where_para1).sum(axis=(2, 3))).dot(what_para1) + np.random.randn(num_sample, 1)
    print(vox1.shape)
    # voxel 2 parameters
    where_para2 = ReceptiveFieldProcessor(vf_size=20, center_x=3,
                                          center_y=5, rf_size=3.5).get_spatial_kernel(kernel_size=resolution)
    where_para2 = where_para2.astype(np.float32)
    what_para2 = ((4 * fw + 1) * (np.abs(4 * fw) < 4)).reshape(num_feature, 1)
    vox2 = ((X_all * where_para2).sum(axis=(2, 3))).dot(what_para2) + np.random.randn(num_sample, 1)
    print(vox2.shape)
    # concatenate vox1 & vox2
    y = np.hstack((vox1, vox2)).astype(np.float32)
    print(y.shape)

    X_train = X_all[:int(num_sample * 0.85), :]
    y_train = y[:int(num_sample * 0.85), :]
    X_test = X_all[int(num_sample * 0.85):, :]
    y_test = y[int(num_sample * 0.85):, :]


    # ===================model usage==============================
    model = FeatureWightedModel(vf_size=20)
    # model.set_grid_params(center_x=np.linspace(-9, 9, 2), center_y=np.linspace(-9, 9, 2), rf_size=[0.8, 3.6])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = np.sqrt((y_test - y_pred)**2).mean(axis=0)
    print(mse)

    import matplotlib.pyplot as plt

    plt.scatter(model.predict(X_train)[:, 0], y_train[:, 0], color='b')
    plt.scatter(model.predict(X_test)[:, 0], y_test[:, 0], color='r')
    plt.show()
