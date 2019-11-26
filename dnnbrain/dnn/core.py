import PIL
import torch
import numpy as np

from copy import deepcopy
from dnnbrain.io import fileio as fio
from dnnbrain.dnn.base import dnn_mask, dnn_fe, array_statistic
from dnnbrain.dnn.base import UnivariatePredictionModel, MultivariatePredictionModel
from dnnbrain.brain.algo import convolve_hrf


class Stimulus:
    """
    Store and handle stimulus-related information
    """
    def __init__(self, fname=None):
        """
        Parameter:
        ---------
        fname[str]: file name with suffix as .stim.csv
        """
        self.meta = dict()
        self._data = dict()
        if fname is not None:
            self.load(fname)

    def load(self, fname):
        """
        Load stimulus-related information

        Parameter:
        ---------
        fname[str]: file name with suffix as .stim.csv
        """
        stim_file = fio.StimulusFile(fname)
        stimuli = stim_file.read()
        self._data = stimuli.pop('data')
        self.meta = stimuli

    def save(self, fname):
        """
        Save stimulus-related information

        Parameter:
        ---------
        fname[str]: file name with suffix as .stim.csv
        """
        stim_file = fio.StimulusFile(fname)
        meta = self.meta.copy()
        stim_file.write(meta.pop('type'), meta.pop('path'),
                        self._data, **meta)

    def get(self, item):
        """
        Get a column of data according to the item

        Parameter:
        ---------
        item[str]: item name of each column

        Return:
        ------
        col[array]: a column of data
        """
        return self._data[item]

    def set(self, item, value):
        """
        Set a column of data according to the item

        Parameters:
        ----------
        item[str]: item name of the column
        value[array_like]: an array_like data with shape as (n_stim,)
        """
        self._data[item] = np.asarray(value)

    def delete(self, item):
        """
        Delete a column of data according to item

        Parameter:
        ---------
        item[str]: item name of each column
        """
        self._data.pop(item)

    @property
    def items(self):
        return list(self._data.keys())

    def __len__(self):
        """
        the length of the Stimulus object

        Return:
        ------
            [int]: the number of stimulus IDs
        """
        return len(self._data['stimID'])

    def __getitem__(self, indices):
        """
        Get part of the Stimulus object by imitating 2D array's subscript index

        Parameter:
        ---------
        indices[int|list|tuple|slice]: subscript indices

        Return:
        ------
        stim[Stimulus]: a part of the self.
        """
        # parse subscript indices
        if isinstance(indices, int):
            # regard it as row index
            # get all columns
            rows = [indices]
            cols = self.items
        elif isinstance(indices, (slice, list)):
            # regard it all as row indices
            # get all columns
            rows = indices
            cols = self.items
        elif isinstance(indices, tuple):
            if len(indices) == 0:
                # get all rows and columns
                rows = slice(None, None, None)
                cols = self.items
            elif len(indices) == 1:
                # regard the only element as row indices
                # get all columns
                if isinstance(indices[0], int):
                    # regard it as row index
                    rows = [indices[0]]
                elif isinstance(indices[0], (slice, list)):
                    # regard it all as row indices
                    rows = indices[0]
                else:
                    raise IndexError("only integer, slices (`:`), list are valid row indices")
                cols = self.items
            elif len(indices) == 2:
                # regard the first element as row indices
                # regard the second element as column indices
                rows, cols = indices
                if isinstance(rows, int):
                    # regard it as row index
                    rows = [rows]
                elif isinstance(rows, (slice, list)):
                    # regard it all as row indices
                    pass
                else:
                    raise IndexError("only integer, slices (`:`), list are valid row indices")

                if isinstance(cols, int):
                    # get a column according to an integer
                    cols = [self.items[cols]]
                elif isinstance(cols, str):
                    # get a column according to an string
                    cols = [cols]
                elif isinstance(cols, list):
                    if np.all([isinstance(i, int) for i in cols]):
                        # get columns according to a list of integers
                        cols = [self.items[i] for i in cols]
                    elif np.all([isinstance(i, str) for i in cols]):
                        # get columns according to a list of strings
                        pass
                    else:
                        raise IndexError("only integer [list], string [list] and slices (`:`) "
                                         "are valid column indices")
                elif isinstance(cols, slice):
                    # get columns according to a slice
                    cols = self.items[cols]
                else:
                    raise IndexError("only integer [list], string [list] and slices (`:`) "
                                     "are valid column indices")
            else:
                raise IndexError("This is a 2D data, "
                                 "and can't support more than 3 subscript indices!")
        else:
            raise IndexError("only integer, slices (`:`), list and tuple are valid indices")

        # get part of self
        stim = Stimulus()
        stim.meta = self.meta.copy()
        for item in cols:
            stim.set(item, self.get(item)[rows])

        return stim


class Activation:
    """DNN activation"""

    def __init__(self, fname=None, dmask=None):
        """
        Parameters:
        ----------
        fname[str]: DNN activation file
        dmask[Mask]: The mask includes layers/channels/rows/columns of interest.
        """
        self._activation = dict()
        if fname is not None:
            self.load(fname, dmask)

    def load(self, fname, dmask=None):
        """
        Load DNN activation

        Parameters:
        ----------
        fname[str]: DNN activation file
        dmask[Mask]: The mask includes layers/channels/rows/columns of interest.
        """
        if dmask is not None:
            dmask_dict = dict()
            for layer in dmask.layers:
                dmask_dict[layer] = dmask.get(layer)
        else:
            dmask_dict = None

        self._activation = fio.ActivationFile(fname).read(dmask_dict)

    def save(self, fname):
        """
        Save DNN activation

        Parameter:
        ---------
        fname[str]: output file of DNN activation
        """
        fio.ActivationFile(fname).write(self._activation)

    def get(self, layer):
        """
        Get DNN activation

        Parameter:
        ---------
        layer[str]: layer name

        Return:
        ------
            [array]: (n_stim, n_chn, n_row, n_col) array
        """
        return self._activation[layer]

    def set(self, layer, data):
        """
        Set DNN activation

        Parameters:
        ----------
        layer[str]: layer name
        data[array]: 4D DNN activation array with shape (n_stim, n_chn, n_row, n_col)
        """
        self._activation[layer] = data

    def delete(self, layer):
        """
        Delete DNN activation

        Parameter:
        ---------
        layer[str]: layer name
        """
        self._activation.pop(layer)

    def concatenate(self, activations):
        """
        Concatenate activations from different batches of stimuli

        Parameter:
        ---------
        activations[list]: a list of Activation objects

        Return:
        ------
        activation[Activation]: DNN activation
        """
        # check availability
        for i, v in enumerate(activations, 1):
            if not isinstance(v, Activation):
                raise TypeError('All elements in activations must be instances of Activation!')
            if sorted(self.layers) != sorted(v.layers):
                raise ValueError("The element{}'s layers mismatch with self!".format(i))

        # concatenate
        activation = Activation()
        for layer in self.layers:
            # concatenate activation
            data = [v.get(layer) for v in activations]
            data.insert(0, self.get(layer))
            data = np.concatenate(data)
            activation.set(layer, data)

        return activation

    @property
    def layers(self):
        return list(self._activation.keys())

    def mask(self, dmask):
        """
        Mask DNN activation

        Parameter:
        ---------
        dmask[Mask]: The mask includes layers/channels/rows/columns of interest.

        Return:
        ------
        activation[Activation]: DNN activation
        """
        activation = Activation()
        for layer in dmask.layers:
            mask = dmask.get(layer)
            data = dnn_mask(self.get(layer), mask.get('chn'),
                            mask.get('row'), mask.get('col'))
            activation.set(layer, data)

        return activation

    def pool(self, method):
        """
        Pooling DNN activation for each channel

        Parameter:
        ---------
        method[str]: pooling method, choices=(max, mean, median, L1, L2)

        Return:
        ------
        activation[Activation]: DNN activation
        """
        activation = Activation()
        for layer, data in self._activation.items():
            data = array_statistic(data, method, (2, 3), True)
            activation.set(layer, data)

        return activation

    def fe(self, method, n_feat, axis=None):
        """
        Extract features of DNN activation

        Parameters:
        ----------
        method[str]: feature extraction method, choices=(pca, hist, psd)
            pca: use n_feat principal components as features
            hist: use histogram of activation as features
                Note: n_feat equal-width bins in the given range will be used!
                psd: use power spectral density as features
        n_feat[int]: The number of features to extract
        axis{str}: axis for feature extraction, choices=(chn, row_col)

        Return:
        ------
        activation[Activation]: DNN activation
        """
        activation = Activation()
        for layer, data in self._activation.items():
            data = dnn_fe(data, method, n_feat, axis)
            activation.set(layer, data)

        return activation

    def convolve_hrf(self, onsets, durations, n_vol, tr, ops=100):
        """
        Convolve DNN activation with HRF and align with the timeline of BOLD signal

        Parameters:
        ----------
        onsets[array_like]: in sec. size = n_event
        durations[array_like]: in sec. size = n_event
        n_vol[int]: the number of volumes of BOLD signal
        tr[float]: repeat time in second
        ops[int]: oversampling number per second

        Return:
        ------
        activation[Activation]: DNN activation
        """
        activation = Activation()
        for layer, data in self._activation.items():
            n_stim, n_chn, n_row, n_col = data.shape
            data = convolve_hrf(data.reshape(n_stim, -1), onsets, durations,
                                n_vol, tr, ops)
            data = data.reshape(n_vol, n_chn, n_row, n_col)
            activation.set(layer, data)

        return activation

    def _check_arithmetic(self, other):
        """
        Check availability of the arithmetic operation for self

        Parameter:
        ---------
        other[Activation]: DNN activation
        """
        if not isinstance(other, Activation):
            raise TypeError("unsupported operand type(s): "
                            "'{0}' and '{1}'".format(type(self), type(other)))
        assert sorted(self.layers) == sorted(other.layers), \
            "The two object's layers mismatch!"
        for layer in self.layers:
            assert self.get(layer).shape == other.get(layer).shape, \
                "{}'s activation shape mismatch!".format(layer)

    def __add__(self, other):
        """
        Define addition operation

        Parameter:
        ---------
        other[Activation]: DNN activation

        Return:
        ------
        activation[Activation]: DNN activation
        """
        self._check_arithmetic(other)

        activation = Activation()
        for layer in self.layers:
            data = self.get(layer) + other.get(layer)
            activation.set(layer, data)

        return activation

    def __sub__(self, other):
        """
        Define subtraction operation

        Parameter:
        ---------
        other[Activation]: DNN activation

        Return:
        ------
        activation[Activation]: DNN activation
        """
        self._check_arithmetic(other)

        activation = Activation()
        for layer in self.layers:
            data = self.get(layer) - other.get(layer)
            activation.set(layer, data)

        return activation

    def __mul__(self, other):
        """
        Define multiplication operation

        Parameter:
        ---------
        other[Activation]: DNN activation

        Return:
        ------
        activation[Activation]: DNN activation
        """
        self._check_arithmetic(other)

        activation = Activation()
        for layer in self.layers:
            data = self.get(layer) * other.get(layer)
            activation.set(layer, data)

        return activation

    def __truediv__(self, other):
        """
        Define true division operation

        Parameter:
        ---------
        other[Activation]: DNN activation

        Return:
        ------
        activation[Activation]: DNN activation
        """
        self._check_arithmetic(other)

        activation = Activation()
        for layer in self.layers:
            data = self.get(layer) / other.get(layer)
            activation.set(layer, data)

        return activation

    def __getitem__(self, indices):
        """
        Get part of Activation along stimulus axis

        Parameter:
        ---------
        indices[int|list|slice]: indices of stimulus axis

        Return:
        ------
        activation[Activation]: DNN activation
        """
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, (list, slice)):
            pass
        else:
            raise IndexError("only integer, slices (`:`), and list are valid indices")

        activation = Activation()
        for layer in self.layers:
            data = self.get(layer)[indices]
            activation.set(layer, data)

        return activation


class Mask:
    """DNN mask"""

    def __init__(self, fname=None):
        """
        Parameter:
        ---------
        fname[str]: DNN mask file
        """
        self._dmask = dict()
        if fname is not None:
            self.load(fname)

    def load(self, fname):
        """
        Load DNN mask, the whole mask will be overrode.

        Parameter:
        ---------
        fname[str]: DNN mask file
        """
        self._dmask = fio.MaskFile(fname).read()

    def save(self, fname):
        """
        Save DNN mask

        Parameter:
        ---------
        fname[str]: output file name of DNN mask
        """
        fio.MaskFile(fname).write(self._dmask)

    def get(self, layer):
        """
        Get mask of a layer

        Parameter:
        ---------
        layer[str]: layer name

        Return:
        ------
            [dict]: layer mask
        """
        return self._dmask[layer]

    def set(self, layer, channels=None, rows=None, columns=None):
        """
        Set DNN mask.

        Parameters:
        ----------
        layer[str]: layer name
        channels[list]: sequence numbers of channels of interest.
        rows[list]: sequence numbers of rows of interest.
        columns[list]: sequence numbers of columns of interest.
        """
        if layer not in self._dmask:
            self._dmask[layer] = dict()
        if channels is not None:
            assert isinstance(channels, list), "'channels' must be a list!"
            self._dmask[layer]['chn'] = channels
        if rows is not None:
            assert isinstance(rows, list), "'rows' must be a list!"
            self._dmask[layer]['row'] = rows
        if columns is not None:
            assert isinstance(columns, list), "'columns' must be a list!"
            self._dmask[layer]['col'] = columns

    def copy(self):
        """
        Make a copy of the DNN mask

        Return:
        ------
        dmask[Mask]: The mask includes layers/channels/rows/columns of interest.
        """
        dmask = Mask()
        dmask._dmask = deepcopy(self._dmask)

        return dmask

    def delete(self, layer):
        """
        Delete a layer

        Parameter:
        ---------
        layer[str]: layer name
        """
        self._dmask.pop(layer)

    def clear(self):
        """
        Empty the DNN mask
        """
        self._dmask.clear()

    @property
    def layers(self):
        return list(self._dmask.keys())


class Image:
    """
    The Image class encapsulate an PIL.Image object, and provides
    some methods to load/save/operate the Image object.
    """

    def __init__(self, data=None):
        """
        Parameter:
        ---------
        data[PIL.Image|ndarray|Tensor]: image data
            If is PIL.Image, it will be converted to RGB mode.
                As a result, nothing is changed if it is 'RGB' mode originally, while pixel
                value's type is converted to unsigned int8 when it is 'F' mode.
                Note: The two newly increased channels when convert 'L'/'F' to 'RGB'
                    mode is just two copies of the original channel.
            If is ndarray|Tensor, its shape must be ([3, ]height, width).
                If the shape is (height, width), it is a gray image.
                    And we will expand two copies of itself to match
                    the shape with 'RGB' mode, i.e. (3, height, width).
                If the shape is (3, height, width), it is a color image.
        """
        self.data = None
        if data is not None:
            self.set(data)

    def load(self, fname):
        """
        Load Image from file.

        Parameter:
        ---------
        fname[str]: image file name
        """
        self.data = PIL.Image.open(fname).convert('RGB')

    def save(self, fname):
        """
        Save Image to file.

        Parameter:
        ---------
        fname[str]: image file name
        """
        self.data.save(fname)

    def get(self):
        """
        Get image data as PIL.Image

        Return:
        ------
        data[PIL.Image]: image data
        """
        return self.data

    def set(self, data):
        """
        Set image data from data

        Parameter:
        ---------
        data[PIL.Image|ndarray|Tensor]: image data
            If is PIL.Image.Image, it will be converted to RGB mode.
                As a result, nothing is changed if it is 'RGB' mode originally, while pixel
                value's type is converted to unsigned int8 when it is 'F' mode.
                Note: The two newly increased channels when convert 'L'/'F' to 'RGB'
                    mode is just two copies of the original channel.
            If is ndarray|Tensor, its shape must be ([3, ]height, width).
                If the shape is (height, width), it is a gray image.
                    And we will expand two copies of itself to match
                    the shape with 'RGB' mode, i.e. (3, height, width).
                If the shape is (3, height, width), it is a color image.
        """
        if isinstance(data, PIL.Image.Image):
            self.data = data.convert('RGB')

        elif isinstance(data, (np.ndarray, torch.Tensor)):
            # deal with data type
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            assert data.dtype is np.dtype('uint8'), "Pixel value's dtype must be unsigned int8 in 'RGB' mode!"

            # deal with data shape
            if data.ndim == 2:
                data = np.expand_dims(data, 0)
                data = np.repeat(data, 3, 0)
            elif data.ndim == 3:
                assert data.shape[0] == 3, 'Color image has and only has 3 channels!'
            else:
                raise ValueError("The shape of ndarray|Tensor must be ([3, ]height, width)")

            # set image list
            self.data = PIL.Image.fromarray(data.transpose((1, 2, 0)))

        else:
            raise TypeError("Only PIL.Image, ndarray and Tensor are supported!"
                            f"Get {type(data)} instead.")

    def to_array(self):
        """
        Get image data as numpy array

        Return:
        ------
        data[ndarray]: image data with shape as (3, height, width)
        """
        data = [np.asarray(img) for img in self.data]
        data = np.asarray(data).transpose((0, 3, 1, 2))

        return data

    def to_tensor(self):
        """
        Get image data as torch tensor

        Return:
        ------
        data[Tensor]: image data with shape as (3, height, width)
        """
        data = self.to_array()
        data = torch.tensor(data)

        return data

    def resize(self, size, resample=0):
        """
        Resize Image

        Parameters:
        ----------
        size[tuple]: the target size
            as a 2-tuple: (width, height)
        resample[int]: resampling filter, for example:
            Image.NEAREST == 0
            Image.BILINEAR == 2
            Image.HAMMING == 5
        """
        self.data = self.data.resize(size, resample)

    def crop(self, box):
        """
        Crop Image with a rectangular region

        Parameter:
        ---------
        box[tuple]: the crop rectangle
            as a (left, upper, right, lower)-tuple
        """
        self.data = self.data.crop(box)


class DnnProbe:
    """
    Decode DNN activation to behavior data. As a result,
    probe the ability of DNN activation to predict the behavior.
    """
    def __init__(self, dnn_activ=None, model_type=None, model_name=None, cv=3):
        """
        Parameters:
        ----------
        dnn_activ[Activation]: DNN activation
        model_type[str]: choices=(uv, mv)
            'uv': univariate prediction model
            'mv': multivariate prediction model
        model_name[str]: name of a model used to do prediction
            If is 'corr', it just uses correlation rather than prediction.
                And the model_type must be 'uv'.
        cv[int]: cross validation fold number
        """
        self.set(dnn_activ, model_type, model_name, cv)

    def set(self, dnn_activ=None, model_type=None, model_name=None, cv=None):
        """
        Set some attributes

        Parameters:
        ----------
        dnn_activ[Activation]: DNN activation
        model_type[str]: choices=(uv, mv)
            'uv': univariate prediction model
            'mv': multivariate prediction model
        model_name[str]: name of a model used to do prediction
            If is 'corr', it just uses correlation rather than prediction.
                And the model_type must be 'uv'.
        cv[int]: cross validation fold number
        """
        if dnn_activ is not None:
            self.dnn_activ = dnn_activ

        if model_type is None:
            pass
        elif model_type == 'uv':
            self.model = UnivariatePredictionModel()
        elif model_type == 'mv':
            self.model = MultivariatePredictionModel()
        else:
            raise ValueError('model_type must be one of the (uv, mv).')

        if model_name is not None:
            if not hasattr(self, 'model'):
                raise RuntimeError('You have to set model_type first!')
            self.model.set(model_name)

        if cv is not None:
            if not hasattr(self, 'model'):
                raise RuntimeError('You have to set model_type first!')
            self.model.set(cv=cv)

    def probe(self, beh_data, iter_axis=None):
        """
        Probe the ability of DNN activation to predict the behavior.

        Parameters:
        ----------
        beh_data[ndarray]: behavior data with shape as (n_stim, n_beh)
        iter_axis[str]: iterate along the specified axis
            ---for uv---
            channel: Summarize the maximal prediction score for each channel.
            row_col: Summarize the maximal prediction score for each position (row_idx, col_idx).
            default: Summarize the maximal prediction score for the whole layer.
            ---for mv---
            channel: Do multivariate prediction using all units in each channel.
            row_col: Do multivariate prediction using all units in each position (row_idx, col_idx).
            default: Do multivariate prediction using all units in the whole layer.

        Return:
        ------
        pred_dict[dict]:
            ---for uv---
            layer:
                score[ndarray]: max scores
                    shape=(n_iter, n_beh)
                model[ndarray]: fitted models of the max scores
                    shape=(n_iter, n_beh)
                chn_loc[ndarray]: channel locations of the max scores
                    shape=(n_iter, n_beh)
                row_loc[ndarray]: row locations of the max scores
                    shape=(n_iter, n_beh)
                col_loc[ndarray]: column locations of the max scores
                    shape=(n_iter, n_beh)
            ---for mv---
            layer:
                score[ndarray]: prediction scores
                    shape=(n_iter, n_beh)
                model[ndarray]: fitted models
                    shape=(n_iter, n_beh)
        """
        _, n_beh = beh_data.shape

        pred_dict = dict()
        for layer in self.dnn_activ.layers:
            # get DNN activation and reshape it to 3D
            activ = self.dnn_activ.get(layer)
            n_stim, n_chn, n_row, n_col = activ.shape
            n_row_col = n_row * n_col
            activ = activ.reshape((n_stim, n_chn, n_row_col))

            # transpose axis to make activ's shape as (n_stimulus, n_iterator, n_element)
            if iter_axis is None:
                activ = activ.reshape((n_stim, 1, -1))
            elif iter_axis == 'row_col':
                activ = activ.transpose((0, 2, 1))
            elif iter_axis == 'channel':
                pass
            else:
                raise ValueError("Unsupported iter_axis:", iter_axis)
            n_stim, n_iter, n_elem = activ.shape

            # start encoding
            score_arr = np.zeros((n_iter, n_beh), dtype=np.float)
            model_arr = np.zeros_like(score_arr, dtype=np.object)
            if isinstance(self.model, UnivariatePredictionModel):
                channel_arr = np.zeros_like(score_arr, dtype=np.int)
                row_arr = np.zeros_like(score_arr, dtype=np.int)
                column_arr = np.zeros_like(score_arr, dtype=np.int)
                for iter_idx in range(n_iter):
                    data = self.model.predict(activ[:, iter_idx, :], beh_data)
                    score_arr[iter_idx] = data['score']
                    model_arr[iter_idx] = data['model']
                    for beh_idx, loc in enumerate(data['location']):
                        # transform location
                        if iter_axis is None:
                            chn_idx = loc // n_row_col
                            row_idx = loc % n_row_col // n_col
                            col_idx = loc % n_row_col % n_col
                        elif iter_axis == 'channel':
                            chn_idx = iter_idx
                            row_idx = loc // n_col
                            col_idx = loc % n_col
                        else:
                            chn_idx = loc
                            row_idx = iter_idx // n_col
                            col_idx = iter_idx % n_col

                        channel_arr[iter_idx, beh_idx] = chn_idx + 1
                        row_arr[iter_idx, beh_idx] = row_idx + 1
                        column_arr[iter_idx, beh_idx] = col_idx + 1
                    print(f'Layer-{layer} iter-{iter_idx+1}/{n_iter}')

                pred_dict[layer] = {
                    'score': score_arr,
                    'model': model_arr,
                    'chn_loc': channel_arr,
                    'row_loc': row_arr,
                    'col_loc': column_arr
                }
            else:
                for iter_idx in range(n_iter):
                    data = self.model.predict(activ[:, iter_idx, :], beh_data)
                    score_arr[iter_idx] = data['score']
                    model_arr[iter_idx] = data['model']
                    print(f'Layer-{layer} iter-{iter_idx+1}/{n_iter}')

                pred_dict[layer] = {
                    'score': score_arr,
                    'model': model_arr
                }
        return pred_dict
