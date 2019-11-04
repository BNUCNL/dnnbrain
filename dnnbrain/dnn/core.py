import torch
import numpy as np

from copy import deepcopy
import dnnbrain.io.fileio as iofile
from dnnbrain.dnn.base import DNNLoader
from dnnbrain.dnn.base import array_statistic
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
from scipy.signal import convolve, periodogram
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


class Stimulus:
    """
    Store and handle stimulus-related information
    """
    def __init__(self, path=None):
        """
        Parameter:
        ---------
        path[str]: file path with suffix as .stim.csv
        """
        self.meta = dict()
        self._data = dict()
        if path is not None:
            self.load(path)

    def load(self, path):
        """
        Load stimulus-related information

        Parameter:
        ---------
        path[str]: file path with suffix as .stim.csv
        """
        stim_file = iofile.StimulusFile(path)
        stimuli = stim_file.read()
        self._data = stimuli.pop('data')
        self.meta = stimuli

    def save(self, path):
        """
        Save stimulus-related information

        Parameter:
        ---------
        path[str]: file path with suffix as .stim.csv
        """
        stim_file = iofile.StimulusFile(path)
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

    def __getitem__(self, indices):
        """
        Get part of the Stimulus object by imitating 2D array's subscript index

        Parameter:
        ---------
        indices[int|tuple|slice]: subscript indices

        Return:
        ------
        stim[Stimulus]: a part of the self.
        """
        # parse subscript indices
        if isinstance(indices, (int, slice, list)):
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
                rows = indices[0]
                cols = self.items
            elif len(indices) == 2:
                # regard the first element as row indices
                # regard the second element as column indices
                rows, cols = indices
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


class DNN:
    """Deep neural network"""

    def __init__(self, net=None):
        """
        Parameter:
        ---------
        net[str]: deep neural network name
        """
        self.model = None
        self.layer2loc = None
        self.img_size = None
        if net is not None:
            self.load(net)

    def load(self, net):
        """
        Load DNN and associated information

        Parameter:
        ---------
        net[str]: deep neural network name
        """
        loader = DNNLoader(net)
        self.model = loader.model
        self.layer2loc = loader.layer2loc
        self.img_size = loader.img_size

    def save(self, path):
        """
        Save DNN parameters

        Parameter:
        ---------
        path[str]: output file path with suffix as .pth
        """
        assert path.endswith('.pth'), 'File suffix must be .pth'
        torch.save(self.model.state_dict(), path)

    def compute_activation(self, data, dmask):
        """
        Extract DNN activation

        Parameters:
        ----------
        data[tensor]: input stimuli of the model with shape as (n_stim, n_chn, height, width)
        dmask[Mask]: The mask includes layers/channels/columns of interest.

        Return:
        ------
        act[Activation]: DNN activation
        """
        # change to eval mode
        self.model.eval()

        act = Activation()
        for layer in dmask.layers:
            # prepare dnn activation hook
            acts_holder = []

            def hook_act(module, input, output):
                # copy dnn activation and record raw shape
                acts = output.detach().numpy().copy()
                raw_shape = acts.shape

                # reshape dnn activation and mask it
                acts = acts.reshape((raw_shape[0], raw_shape[1], -1))
                acts = dnn_mask(acts, dmask.get(layer, 'chn'),
                                dmask.get(layer, 'col'))

                # hold the information
                acts_holder.append(acts)
                acts_holder.append(raw_shape)

            module = self.model
            for k in self.layer2loc[layer]:
                module = module._modules[k]
            hook_handle = module.register_forward_hook(hook_act)

            # extract dnn activation
            self.model(data)
            act.set(layer, *acts_holder)
            hook_handle.remove()

        return act

    def get_kernel(self, layer, kernel_num=None):
        """
        Get kernel's weights of the layer

        Parameters:
        ----------
        layer[str]: layer name
        kernel_num[int]: the sequence number of the kernel

        Return:
        ------
        kernel[array]: kernel weights
        """
        # localize the module
        module = self.model
        for k in self.layer2loc[layer]:
            module = module._modules[k]

        # get the weights
        kernel = module.weight
        if kernel_num is not None:
            kernel = kernel[kernel_num]

        return kernel.detach().numpy()

    def ablate(self, layer, channels=None):
        """
        Ablate DNN kernels' weights

        Parameters:
        ----------
        layer[str]: layer name
        channels[list]: sequence numbers of channels of interest
            If None, ablate the whole layer.
        """
        # localize the module
        module = self.model
        for k in self.layer2loc[layer]:
            module = module._modules[k]

        # ablate kernels' weights
        if channels is None:
            module.weight.data[:] = 0
        else:
            channels = [chn - 1 for chn in channels]
            module.weight.data[channels] = 0


class Activation:
    """DNN activation"""

    def __init__(self, path=None, dmask=None):
        """
        Parameters:
        ----------
        path[str]: DNN activation file
        dmask[Mask]: The mask includes layers/channels/columns of interest.
        """
        self._act = dict()
        if path is not None:
            self.load(path, dmask)

    def load(self, path, dmask=None):
        """
        Load DNN activation

        Parameters:
        ----------
        path[str]: DNN activation file
        dmask[Mask]: The mask includes layers/channels/columns of interest.
        """
        if dmask is not None:
            dmask = dmask._mask
        self._act = iofile.ActivationFile(path).read(dmask)

    def save(self, path):
        """
        Save DNN activation

        Parameter:
        ---------
        path[str]: output file of DNN activation
        """
        iofile.ActivationFile(path).write(self._act)

    def get(self, layer, raw_shape=False):
        """
        Get DNN activation or its raw_shape (if exist)

        Parameters:
        ----------
        layer[str]: layer name
        raw_shape[bool]:
            If true, get raw_shape of the layer's activation.
            If false, get the layer's activation.

        Return:
        ------
        data[tuple|array]: raw shape or (n_stim, n_chn, n_col) array
        """
        if raw_shape:
            data = self._act[layer]['raw_shape']
        else:
            data = self._act[layer]['data']

        return data

    def set(self, layer, value=None, raw_shape=None):
        """
        Set DNN activation or its raw shape
        If the layer doesn't exist, initiate it with the value.

        Parameters:
        ----------
        layer[str]: layer name
        value[array]: 3D DNN activation array with shape (n_stim, n_chn, n_col)
        raw_shape[tuple]: raw_shape of the layer's activation
        """
        if layer not in self._act:
            if value is None:
                raise ValueError("The value can't be None when initiating a new layer!")
            self._act[layer] = {'data': value, 'raw_shape': ()}
        else:
            if value is not None:
                self._act[layer]['data'] = value

        if raw_shape is not None:
            self._act[layer]['raw_shape'] = raw_shape

    def delete(self, layer, raw_shape=False):
        """
        Delete DNN activation or its raw_shape attribution

        Parameters:
        ----------
        layer[str]: layer name
        raw_shape[bool]:
            If true, delete raw_shape of the layer's activation.
            If false, delete the layer's activation and raw_shape.
        """
        if raw_shape:
            self._act[layer].pop('raw_shape')
        else:
            self._act.pop(layer)

    def concatenate(self, acts):
        """
        Concatenate activations from different batches of stimuli

        Parameter:
        ---------
        acts[list]: a list of Activation objects

        Return:
        ------
        act[Activation]: DNN activation
        """
        # check availability
        for i, v in enumerate(acts, 1):
            if not isinstance(v, Activation):
                raise TypeError('All elements in acts must be instances of Activation!')
            if sorted(self.layers) != sorted(v.layers):
                raise ValueError("The element{}'s layers mismatch with self!".format(i))

        # concatenate
        act = Activation()
        for layer in self.layers:
            # concatenate activation
            data = [v.get(layer) for v in acts]
            data.insert(0, self.get(layer))
            data = np.concatenate(data)

            # update raw shape
            n_stim = data.shape[0]
            raw_shape = (n_stim,) + self.get(layer, True)[1:]
            act.set(layer, data, raw_shape)

        return act

    @property
    def layers(self):
        return list(self._act.keys())

    def mask(self, dmask):
        """
        Mask DNN activation

        Parameter:
        ---------
        dmask[Mask]: The mask includes layers/channels/columns of interest.

        Return:
        ------
        act[Activation]: DNN activation
        """
        act = Activation()
        for layer in dmask.layers:
            channels = dmask.get(layer, 'chn')
            columns = dmask.get(layer, 'col')
            data = dnn_mask(self.get(layer), channels, columns)
            act.set(layer, data, self.get(layer, True))

        return act

    def pool(self, method, dmask=None):
        """
        Pooling DNN activation for each channel

        Parameters:
        ----------
        method[str]: pooling method, choices=(max, mean, median)
        dmask[Mask]: The mask includes layers/channels/columns of interest.

        Return:
        ------
        act[Activation]: DNN activation
        """
        act = Activation()
        if dmask is None:
            for layer, d in self._act.items():
                data = dnn_pooling(d['data'], method)
                act.set(layer, data, d['raw_shape'])
        else:
            for layer in dmask.layers:
                channels = dmask.get(layer, 'chn')
                columns = dmask.get(layer, 'col')
                data = dnn_mask(self.get(layer), channels, columns)
                data = dnn_pooling(data, method)
                act.set(layer, data, self.get(layer, True))

        return act

    def fe(self, method, n_feature, axis=None, dmask=None):
        """
        Extract features of DNN activation

        Parameters:
        ----------
        method[str]: feature extraction method, choices=(pca, hist, psd)
            pca: use n_feature principal components as features
            hist: use histogram of activation as features
                Note: n_feature equal-width bins in the given range will be used!
                psd: use power spectral density as features
        n_feature[int]: The number of features to extract
        axis{str}: axis for feature extraction, choices=(chn, col)
            If it's None, extract features from the whole layer. Note:
            The result of this will be an array with shape (n_stim, n_feat, 1), but
            We also regard it as (n_stim, n_chn, n_col)
        dmask[Mask]: The mask includes layers/channels/columns of interest.

        Returns:
        -------
        act[Activation]: DNN activation
        """
        act = Activation()
        if dmask is None:
            for layer, d in self._act.items():
                data = dnn_fe(d['data'], method, n_feature, axis)
                act.set(layer, data, d['raw_shape'])
        else:
            for layer in dmask.layers:
                channels = dmask.get(layer, 'chn')
                columns = dmask.get(layer, 'col')
                data = dnn_mask(self.get(layer), channels, columns)
                data = dnn_fe(data, method, n_feature, axis)
                act.set(layer, data, self.get(layer, True))

        return act

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
            assert self.get(layer, True) == other.get(layer, True), \
                "{}'s raw shape mismatch!".format(layer)

    def __add__(self, other):
        """
        Define addition operation

        Parameter:
        ---------
        other[Activation]: DNN activation

        Return:
        ------
        act[Activation]: DNN activation
        """
        self._check_arithmetic(other)

        act = Activation()
        for layer in self.layers:
            data = self.get(layer) + other.get(layer)
            act.set(layer, data, self.get(layer, True))

        return act

    def __sub__(self, other):
        """
        Define subtraction operation

        Parameter:
        ---------
        other[Activation]: DNN activation

        Return:
        ------
        act[Activation]: DNN activation
        """
        self._check_arithmetic(other)

        act = Activation()
        for layer in self.layers:
            data = self.get(layer) - other.get(layer)
            act.set(layer, data, self.get(layer, True))

        return act

    def __mul__(self, other):
        """
        Define multiplication operation

        Parameter:
        ---------
        other[Activation]: DNN activation

        Return:
        ------
        act[Activation]: DNN activation
        """
        self._check_arithmetic(other)

        act = Activation()
        for layer in self.layers:
            data = self.get(layer) * other.get(layer)
            act.set(layer, data, self.get(layer, True))

        return act

    def __truediv__(self, other):
        """
        Define true division operation

        Parameter:
        ---------
        other[Activation]: DNN activation

        Return:
        ------
        act[Activation]: DNN activation
        """
        self._check_arithmetic(other)

        act = Activation()
        for layer in self.layers:
            data = self.get(layer) / other.get(layer)
            act.set(layer, data, self.get(layer, True))

        return act


class Mask:
    """DNN mask"""

    def __init__(self, path=None):
        """
        Parameter:
        ---------
        path[str]: DNN mask file
        """
        self._mask = dict()
        if path is not None:
            self.load(path)

    def load(self, path):
        """
        Load DNN mask, the whole mask will be overrode.

        Parameter:
        ---------
        path[str]: DNN mask file
        """
        self._mask = iofile.MaskFile(path).read()

    def save(self, path):
        """
        Save DNN mask

        Parameter:
        ---------
        path[str]: output file path of DNN mask
        """
        iofile.MaskFile(path).write(self._mask)

    def get(self, layer, axis=None):
        """
        Get mask of a layer

        Parameters:
        ----------
        layer[str]: layer name
        axis[str]: chn or col

        Return:
        ------
        dmask[dict|list]: layer mask
        """
        dmask = self._mask[layer]
        if axis is not None:
            dmask = dmask[axis]

        return dmask

    def set(self, layer, channels=None, columns=None):
        """
        Set DNN mask.
        If the layer doesn't exist, initiate it with all channels and columns.

        Parameters:
        ----------
        layer[str]: layer name
        channels[list|str]:
            If is list, it contains sequence numbers of channels of interest.
            If is str, it must be 'all' that means all channels in the layer.
            If is None, do nothing.
        columns[list|str]:
            If is list, it contains sequence numbers of columns of interest.
            If is str, it must be 'all' that means all columns in the layer.
            If is None, do nothing.
        """
        if layer not in self._mask:
            self._mask[layer] = {'chn': 'all', 'col': 'all'}
        if channels is not None:
            self._mask[layer]['chn'] = channels
        if columns is not None:
            self._mask[layer]['col'] = columns

    def copy(self):
        """
        Make a copy of the DNN mask
        """
        dmask = Mask()
        dmask._mask = deepcopy(self._mask)

        return dmask

    def delete(self, layer):
        """
        Delete a layer

        Parameter:
        ---------
        layer[str]: layer name
        """
        self._mask.pop(layer)

    def clear(self):
        """
        Empty the DNN mask
        """
        self._mask.clear()

    @property
    def layers(self):
        return list(self._mask.keys())


def dnn_activation(data, model, layer_loc, channels=None):
    """
    Extract DNN activation from the specified layer

    Parameters:
    ----------
    data[tensor]: input stimuli of the model with shape as (n_stim, n_chn, height, width)
    model[model]: DNN model
    layer_loc[sequence]: a sequence of keys to find the location of
        the target layer in the DNN model. For example, the location of the
        fifth convolution layer in AlexNet is ('features', '10').
    channels[list]: channel indices of interest

    Return:
    ------
    dnn_acts[array]: DNN activation
        a 4D array with its shape as (n_stim, n_chn, n_r, n_c)
    """
    # change to eval mode
    model.eval()

    # prepare dnn activation hook
    dnn_acts = []

    def hook_act(module, input, output):
        act = output.detach().numpy().copy()
        if channels is not None:
            act = act[:, channels]
        dnn_acts.append(act)

    module = model
    for k in layer_loc:
        module = module._modules[k]
    hook_handle = module.register_forward_hook(hook_act)

    # extract dnn activation
    model(data)
    dnn_acts = dnn_acts[0]

    hook_handle.remove()
    return dnn_acts


def dnn_mask(dnn_acts, channels='all', columns='all'):
    """
    Extract DNN activation

    Parameters:
    ----------
    dnn_acts[array]: DNN activation, A 3D array with its shape as (n_stim, n_chn, n_col)
    channels[list|str]:
            If is list, it contains sequence numbers of channels of interest.
            If is str, it must be 'all' that means all channels in the layer.
    columns[list|str]:
            If is list, it contains sequence numbers of columns of interest.
            If is str, it must be 'all' that means all columns in the layer.

    Return:
    ------
    dnn_acts[array]: DNN activation after mask
        a 3D array with its shape as (n_stim, n_chn, n_col)
    """
    if channels != 'all':
        channels = [chn-1 for chn in channels]
        dnn_acts = dnn_acts[:, channels, :]
    if columns != 'all':
        columns = [col-1 for col in columns]
        dnn_acts = dnn_acts[:, :, columns]

    return dnn_acts


def dnn_pooling(dnn_acts, method):
    """
    Pooling DNN activation for each channel

    Parameters:
    ------------
    dnn_acts[array]: DNN activation, A 3D array with its shape as (n_stim, n_chn, n_col)
    method[str]: pooling method, choices=(max, mean, median)

    Returns:
    ---------
    dnn_acts[array]: DNN activation after pooling
        a 3D array with its shape as (n_stim, n_chn, 1)
    """
    return array_statistic(dnn_acts, method, 2, True)


def dnn_fe(dnn_acts, meth, n_feat, axis=None):
    """
    Extract features of DNN activation

    Parameters:
    ----------
    dnn_acts[array]: DNN activation
        a 3D array with its shape as (n_stim, n_chn, n_col)
    meth[str]: feature extraction method, choices=(pca, hist, psd)
        pca: use n_feat principal components as features
        hist: use histogram of activation as features
            Note: n_feat equal-width bins in the given range will be used!
        psd: use power spectral density as features
    n_feat[int]: The number of features to extract
    axis{str}: axis for feature extraction, choices=(chn, col)
        If it's None, extract features from the whole layer. Note:
        The result of this will be an array with shape (n_stim, n_feat, 1), but
        We also regard it as (n_stim, n_chn, n_col)

    Returns:
    -------
    dnn_acts_new[array]: DNN activation
        a 3D array with its shape as (n_stim, n_chn, n_col)
    """
    # adjust iterative axis
    n_stim, n_chn, n_col = dnn_acts.shape
    if axis is None:
        dnn_acts = dnn_acts.reshape((n_stim, 1, -1))
    elif axis == 'chn':
        dnn_acts = dnn_acts.transpose((0, 2, 1))
    elif axis == 'col':
        pass
    else:
        raise ValueError('not supported axis:', axis)
    _, n_iter, _ = dnn_acts.shape

    # extract features
    dnn_acts_new = np.zeros((n_stim, n_iter, n_feat))
    if meth == 'pca':
        pca = PCA(n_components=n_feat)
        for i in range(n_iter):
            dnn_acts_new[:, i, :] = pca.fit_transform(dnn_acts[:, i, :])
    elif meth == 'hist':
        for i in range(n_iter):
            for j in range(n_stim):
                dnn_acts_new[j, i, :] = np.histogram(dnn_acts[j, i, :], n_feat)[0]
    elif meth == 'psd':
        for i in range(n_iter):
            for j in range(n_stim):
                f, p = periodogram(dnn_acts[j, i, :])
                dnn_acts_new[j, i, :] = p[:n_feat]
    else:
        raise ValueError('not supported method:', meth)

    # adjust iterative axis
    if axis is None:
        dnn_acts_new = dnn_acts_new.transpose((0, 2, 1))
    elif axis == 'chn':
        dnn_acts_new = dnn_acts_new.transpose((0, 2, 1))

    return dnn_acts_new


def db_uva(dnn_acts, resp, model, iter_axis=None, cvfold=3):
    """
    Use DNN activation to predict responses of brain or behavior
    by univariate analysis.'

    Parameters:
    ----------
    dnn_acts[array]: DNN activation
        A 3D array with its shape as (n_stim, n_chn, n_col)
    resp[array]: response of brain or behavior
        A 2D array with its shape as (n_samp, n_meas)
    model[str]: the name of model used to do prediction
    iter_axis[str]: iterate along the specified axis
        channel: Summarize the maximal prediction score for each channel.
        column: Summarize the maximal prediction score for each column.
        default: Summarize the maximal prediction score for the whole layer.
    cvfold[int]: cross validation fold number

    Return:
    ------
    pred_dict[dict]:
        score_arr: max score array
        channel_arr: channel position of the max score
        column_arr: column position of the max score
        model_arr: fitted model of the max score
    """
    n_stim, n_chn, n_col = dnn_acts.shape
    n_samp, n_meas = resp.shape  # n_sample x n_measures
    assert n_stim == n_samp, 'n_stim != n_samp'

    # transpose axis to make dnn_acts's shape as (n_stimulus, n_iterator, n_element)
    if iter_axis is None:
        dnn_acts = dnn_acts.reshape((n_stim, 1, n_chn * n_col))
    elif iter_axis == 'column':
        dnn_acts = dnn_acts.transpose((0, 2, 1))
    elif iter_axis == 'channel':
        pass
    else:
        raise ValueError("Unspported iter_axis:", iter_axis)
    n_stim, n_iter, n_elem = dnn_acts.shape

    # prepare model
    if model in ('lrc', 'svc'):
        score_evl = 'accuracy'
    elif model in ('glm', 'lasso'):
        score_evl = 'explained_variance'
    else:
        raise ValueError('unsupported model:', model)

    if model == 'lrc':
        model = LogisticRegression()
    elif model == 'svc':
        model = SVC(kernel='linear', C=0.025)
    elif model == 'lasso':
        model = Lasso()
    else:
        model = LinearRegression()

    # prepare container
    score_arr = np.zeros((n_iter, n_meas), dtype=np.float)
    channel_arr = np.zeros_like(score_arr, dtype=np.int)
    column_arr = np.zeros_like(score_arr, dtype=np.int)
    model_arr = np.zeros_like(score_arr, dtype=np.object)

    # start iteration
    for meas_idx in range(n_meas):
        for iter_idx in range(n_iter):
            score_tmp = []
            for elem_idx in range(n_elem):
                cv_scores = cross_val_score(model, dnn_acts[:, iter_idx, elem_idx][:, None],
                                            resp[:, meas_idx], scoring=score_evl, cv=cvfold)
                score_tmp.append(np.mean(cv_scores))

            # find max score
            max_elem_idx = np.argmax(score_tmp)
            max_score = score_tmp[max_elem_idx]
            score_arr[iter_idx, meas_idx] = max_score

            # find position for the max score
            if iter_axis is None:
                chn_idx = max_elem_idx // n_col
                col_idx = max_elem_idx % n_col
            elif iter_axis == 'channel':
                chn_idx, col_idx = iter_idx, max_elem_idx
            else:
                chn_idx, col_idx = max_elem_idx, iter_idx

            channel_arr[iter_idx, meas_idx] = chn_idx + 1
            column_arr[iter_idx, meas_idx] = col_idx + 1

            # fit the max-score model
            model_arr[iter_idx, meas_idx] = model.fit(dnn_acts[:, iter_idx, max_elem_idx][:, None],
                                                      resp[:, meas_idx])
            print('Meas: {0}/{1}; iter:{2}/{3}'.format(meas_idx + 1, n_meas,
                                                       iter_idx + 1, n_iter,))
    pred_dict = {
        'score': score_arr,
        'chn_pos': channel_arr,
        'col_pos': column_arr,
        'model': model_arr
    }
    return pred_dict


def db_mva(dnn_acts, resp, model, iter_axis=None, cvfold=3):
    """
    Use DNN activation to predict responses of brain or behavior
    by multivariate analysis.'

    Parameters:
    ----------
    dnn_acts[array]: DNN activation
        A 3D array with its shape as (n_stim, n_chn, n_col)
    resp[array]: response of brain or behavior
        A 2D array with its shape as (n_samp, n_meas)
    model[str]: the name of model used to do prediction
    iter_axis[str]: iterate along the specified axis
        channel: Do mva using all units in each channel.
        column: Do mva using all units in each column.
        default: Do mva using all units in the whole layer.
    cvfold[int]: cross validation fold number

    Return:
    ------
    pred_dict[dict]:
        score_arr: prediction score array
        model_arr: fitted model
    """
    n_stim, n_chn, n_col = dnn_acts.shape
    n_samp, n_meas = resp.shape  # n_sample x n_measures
    assert n_stim == n_samp, 'n_stim != n_samp'

    # transpose axis to make dnn_acts's shape as (n_stimulus, n_iterator, n_element)
    if iter_axis is None:
        dnn_acts = dnn_acts.reshape((n_stim, 1, n_chn * n_col))
    elif iter_axis == 'column':
        dnn_acts = dnn_acts.transpose((0, 2, 1))
    elif iter_axis == 'channel':
        pass
    else:
        raise ValueError("Unspported iter_axis:", iter_axis)
    n_stim, n_iter, n_elem = dnn_acts.shape

    # prepare model
    if model in ('lrc', 'svc'):
        score_evl = 'accuracy'
    elif model in ('glm', 'lasso'):
        score_evl = 'explained_variance'
    else:
        raise ValueError('unsupported model:', model)

    if model == 'lrc':
        model = LogisticRegression()
    elif model == 'svc':
        model = SVC(kernel='linear', C=0.025)
    elif model == 'lasso':
        model = Lasso()
    else:
        model = LinearRegression()

    score_arr = []
    model_arr = []
    # start iteration
    for iter_idx in range(n_iter):
        # cross validation
        score_tmp = [cross_val_score(model, dnn_acts[:, iter_idx, :], resp[:, i],
                                     scoring=score_evl, cv=cvfold) for i in range(n_meas)]
        score_arr.append(np.array(score_tmp).mean(-1))

        # fit model
        model.fit(dnn_acts[:, iter_idx, :], resp)
        model_arr.append(model)

        print('Finish iteration{0}/{1}'.format(iter_idx + 1, n_iter))
    score_arr = np.array(score_arr)
    model_arr = np.array(model_arr)

    pred_dict = {
        'score': score_arr,
        'model': model_arr
    }
    return pred_dict


def convolve_hrf(X, onsets, durations, n_vol, tr, ops=100):
    """
    Convolve each X's column iteratively with HRF and align with the timeline of BOLD signal

    parameters:
    ----------
    X[array]: [n_event, n_sample]
    onsets[array_like]: in sec. size = n_event
    durations[array_like]: in sec. size = n_event
    n_vol[int]: the number of volumes of BOLD signal
    tr[float]: repeat time in second
    ops[int]: oversampling number per second

    Returns:
    ---------
    X_hrfed[array]: the result after convolution and alignment
    """
    assert np.ndim(X) == 2, 'X must be a 2D array'
    assert X.shape[0] == len(onsets) and X.shape[0] == len(durations), 'The length of onsets and durations should ' \
                                                                       'be matched with the number of events.'
    assert ops in (10, 100, 1000), 'Oversampling rate must be one of the (10, 100, 1000)!'

    # unify the precision
    decimals = int(np.log10(ops))
    onsets = np.round(np.asarray(onsets), decimals=decimals)
    durations = np.round(np.asarray(durations), decimals=decimals)
    tr = np.round(tr, decimals=decimals)

    n_clipped = 0  # the number of clipped time points earlier than the start point of response
    onset_min = onsets.min()
    if onset_min > 0:
        # The earliest event's onset is later than the start point of response.
        # We supplement it with zero-value event to align with the response.
        X = np.insert(X, 0, np.zeros(X.shape[1]), 0)
        onsets = np.insert(onsets, 0, 0, 0)
        durations = np.insert(durations, 0, onset_min, 0)
        onset_min = 0
    elif onset_min < 0:
        print("The earliest event's onset is earlier than the start point of response.\n"
              "We clip the earlier time points after hrf_convolution to align with the response.")
        n_clipped = int(-onset_min * ops)

    # do convolution in batches for trade-off between speed and memory
    batch_size = int(100000 / ops)
    bat_indices = np.arange(0, X.shape[-1], batch_size)
    bat_indices = np.r_[bat_indices, X.shape[-1]]

    vol_t = (np.arange(n_vol) * tr * ops).astype(int)  # compute volume acquisition timing
    n_time_point = int(((onsets + durations).max()-onset_min) * ops)
    X_hrfed = np.zeros([n_vol, 0])
    for idx, bat_idx in enumerate(bat_indices[:-1]):
        X_bat = X[:, bat_idx:bat_indices[idx+1]]
        # generate X raw time course
        X_tc = np.zeros((n_time_point, X_bat.shape[-1]), dtype=np.float32)
        for i, onset in enumerate(onsets):
            onset_start = int(onset * ops)
            onset_end = int(onset_start + durations[i] * ops)
            X_tc[onset_start:onset_end, :] = X_bat[i, :]

        # generate hrf kernel
        hrf = spm_hrf(tr, oversampling=tr*ops)
        hrf = hrf[:, np.newaxis]

        # convolve X raw time course with hrf kernal
        X_tc_hrfed = convolve(X_tc, hrf, method='fft')
        X_tc_hrfed = X_tc_hrfed[n_clipped:, :]

        # downsample to volume timing
        X_hrfed = np.c_[X_hrfed, X_tc_hrfed[vol_t, :]]

        print('hrf convolution: sample {0} to {1} finished'.format(bat_idx+1, bat_indices[idx+1]))

    return X_hrfed
