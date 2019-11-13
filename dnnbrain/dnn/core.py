import numpy as np

from copy import deepcopy
from dnnbrain.io import fileio as fio
from dnnbrain.dnn.base import dnn_mask, dnn_fe, array_statistic
from dnnbrain.dnn.base import Classifier, Regressor
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
            self._dmask[layer]['chn'] = channels
        if rows is not None:
            self._dmask[layer]['row'] = rows
        if columns is not None:
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


class Encoder:
    """
    Encode DNN activation to response of brain or behavior.
    """
    def __init__(self, name=None, iter_axis=None, cv=3):
        """
        Parameters:
        ----------
        name[str]: the name of a model used to do prediction
        iter_axis[str]: iterate along the specified axis
            ---for uva---
            channel: Summarize the maximal prediction score for each channel.
            row_col: Summarize the maximal prediction score for each position (row_idx, col_idx).
            default: Summarize the maximal prediction score for the whole layer.
            ---for mva---
            channel: Do mva using all units in each channel.
            row_col: Do mva using all units in each position (row_idx, col_idx).
            default: Do mva using all units in the whole layer.
        cv[int]: cross validation fold number
        """
        self.model = None
        self.iter_axis = iter_axis
        self.cv = cv
        if name is not None:
            self.set(name)

    def set(self, name=None, iter_axis=None, cv=None):
        """
        Set some attributes

        Parameters:
        ----------
        name[str]: the name of a model used to do prediction
        iter_axis[str]: iterate along the specified axis
            ---for uva---
            channel: Summarize the maximal prediction score for each channel.
            row_col: Summarize the maximal prediction score for each position (row_idx, col_idx).
            default: Summarize the maximal prediction score for the whole layer.
            ---for mva---
            channel: Do mva using all units in each channel.
            row_col: Do mva using all units in each position (row_idx, col_idx).
            default: Do mva using all units in the whole layer.
        cv[int]: cross validation fold number
        """
        if name is None:
            pass
        elif name in ('lrc', 'svc'):
            self.model = Classifier(name)
        elif name in ('glm', 'lasso'):
            self.model = Regressor(name)
        else:
            raise ValueError('unsupported model:', name)

        if iter_axis is not None:
            self.iter_axis = iter_axis

        if cv is not None:
            self.cv = cv

    def uva(self, activation, response):
        """
        Use DNN activation to predict responses of brain or behavior
        by univariate analysis.

        Parameters:
        ----------
        activation[Activation]: DNN activation
        response[array]: responses of brain or behavior
            A 2D array with its shape as (n_sample, n_measurement)

        Return:
        ------
        pred_dict[dict]:
            layer:
                score: max scores array
                channel: channel positions of the max scores
                row: row positions of the max scores
                column: column positions of the max scores
                model: fitted models of the max scores
        """
        n_samp, n_meas = response.shape  # n_sample x n_measures

        pred_dict = dict()
        for layer in activation.layers:
            # get DNN activation and reshape it to 3D
            dnn_acts = activation.get(layer)
            n_stim, n_chn, n_row, n_col = dnn_acts.shape
            assert n_stim == n_samp, 'n_stim != n_samp'
            n_row_col = n_row * n_col
            dnn_acts = dnn_acts.reshape((n_stim, n_chn, n_row_col))

            # transpose axis to make dnn_acts's shape as (n_stimulus, n_iterator, n_element)
            if self.iter_axis is None:
                dnn_acts = dnn_acts.reshape((n_stim, 1, -1))
            elif self.iter_axis == 'row_col':
                dnn_acts = dnn_acts.transpose((0, 2, 1))
            elif self.iter_axis == 'channel':
                pass
            else:
                raise ValueError("Unspported iter_axis:", self.iter_axis)
            n_stim, n_iter, n_elem = dnn_acts.shape

            # prepare container
            score_arr = np.zeros((n_iter, n_meas), dtype=np.float)
            channel_arr = np.zeros_like(score_arr, dtype=np.int)
            row_arr = np.zeros_like(score_arr, dtype=np.int)
            column_arr = np.zeros_like(score_arr, dtype=np.int)
            model_arr = np.zeros_like(score_arr, dtype=np.object)

            # start iteration
            for meas_idx in range(n_meas):
                for iter_idx in range(n_iter):
                    score_tmp = []
                    for elem_idx in range(n_elem):
                        X = dnn_acts[:, iter_idx, elem_idx][:, None]
                        y = response[:, meas_idx]
                        cv_scores = self.model.cross_val_score(X, y, self.cv)
                        score_tmp.append(np.mean(cv_scores))

                    # find max score
                    max_elem_idx = np.argmax(score_tmp)
                    max_score = score_tmp[max_elem_idx]
                    score_arr[iter_idx, meas_idx] = max_score

                    # find position for the max score
                    if self.iter_axis is None:
                        chn_idx = max_elem_idx // n_row_col
                        row_idx = max_elem_idx % n_row_col // n_col
                        col_idx = max_elem_idx % n_row_col % n_col
                    elif self.iter_axis == 'channel':
                        chn_idx = iter_idx
                        row_idx = max_elem_idx // n_col
                        col_idx = max_elem_idx % n_col
                    else:
                        chn_idx = max_elem_idx
                        row_idx = iter_idx // n_col
                        col_idx = iter_idx % n_col

                    channel_arr[iter_idx, meas_idx] = chn_idx + 1
                    row_arr[iter_idx, meas_idx] = row_idx + 1
                    column_arr[iter_idx, meas_idx] = col_idx + 1

                    # fit the max-score model
                    X = dnn_acts[:, iter_idx, max_elem_idx][:, None]
                    y = response[:, meas_idx]
                    model_arr[iter_idx, meas_idx] = self.model.fit(X, y)
                    print('Meas: {0}/{1}; iter:{2}/{3}'.format(meas_idx + 1, n_meas,
                                                               iter_idx + 1, n_iter))
            pred_dict[layer] = {
                'score': score_arr,
                'channel': channel_arr,
                'row': row_arr,
                'column': column_arr,
                'model': model_arr
            }
        return pred_dict

    def mva(self, activation, response):
        """
        Use DNN activation to predict responses of brain or behavior
        by multivariate analysis.'

        Parameters:
        ----------
        activation[Activation]: DNN activation
        response[array]: responses of brain or behavior
            A 2D array with its shape as (n_sample, n_measurement)

        Return:
        ------
        pred_dict[dict]:
            layer:
                score: prediction scores array
                model: fitted models
        """
        n_samp, n_meas = response.shape  # n_sample x n_measures

        pred_dict = dict()
        for layer in activation.layers:
            # get DNN activation and reshape it to 3D
            dnn_acts = activation.get(layer)
            n_stim, n_chn, n_row, n_col = dnn_acts.shape
            assert n_stim == n_samp, 'n_stim != n_samp'
            n_row_col = n_row * n_col
            dnn_acts = dnn_acts.reshape((n_stim, n_chn, n_row_col))

            # transpose axis to make dnn_acts's shape as (n_stimulus, n_iterator, n_element)
            if self.iter_axis is None:
                dnn_acts = dnn_acts.reshape((n_stim, 1, -1))
            elif self.iter_axis == 'row_col':
                dnn_acts = dnn_acts.transpose((0, 2, 1))
            elif self.iter_axis == 'channel':
                pass
            else:
                raise ValueError("Unspported iter_axis:", self.iter_axis)
            n_stim, n_iter, n_elem = dnn_acts.shape

            score_arr = []
            model_arr = []
            # start iteration
            for iter_idx in range(n_iter):
                # cross validation
                X = dnn_acts[:, iter_idx, :]
                score_tmp = [self.model.cross_val_score(X, response[:, i], self.cv)
                             for i in range(n_meas)]
                score_arr.append(np.asarray(score_tmp).mean(-1))

                # fit model
                model_tmp = [self.model.fit(X, response[:, i]) for i in range(n_meas)]
                model_arr.append(model_tmp)

                print('Finish iteration{0}/{1}'.format(iter_idx + 1, n_iter))
            score_arr = np.array(score_arr)
            model_arr = np.array(model_arr)

            pred_dict[layer] = {
                'score': score_arr,
                'model': model_arr
            }
        return pred_dict


class Decoder:
    """
    Decode DNN activation from response of brain or behavior.
    """
    def __init__(self, name=None, cv=3):
        """
        Parameters:
        ----------
        name[str]: the name of a model used to do prediction
        cv[int]: cross validation fold number
        """
        self.model = None
        self.cv = cv
        if name is not None:
            self.set(name)

    def set(self, name=None, cv=None):
        """
        Set some attributes

        Parameters:
        ----------
        name[str]: the name of a model used to do prediction
        cv[int]: cross validation fold number
        """
        if name is None:
            pass
        elif name in ('lrc', 'svc'):
            self.model = Classifier(name)
        elif name in ('glm', 'lasso'):
            self.model = Regressor(name)
        else:
            raise ValueError('unsupported model:', name)

        if cv is not None:
            self.cv = cv

    def uva(self, response, activation):
        """
        Use responses of brain or behavior to predict DNN activation
        by univariate analysis.

        Parameters:
        ----------
        response[array]: responses of brain or behavior
            A 2D array with its shape as (n_sample, n_measurement)
        activation[Activation]: DNN activation

        Return:
        ------
        pred_dict[dict]:
            layer:
                score: max scores array
                measurement: measurement positions of the max scores
                model: fitted models of the max scores
        """
        n_samp, n_meas = response.shape  # n_sample x n_measures

        pred_dict = dict()
        for layer in activation.layers:
            # get DNN activation
            dnn_acts = activation.get(layer)
            n_stim, n_chn, n_row, n_col = dnn_acts.shape
            assert n_stim == n_samp, 'n_stim != n_samp'

            # prepare container
            score_arr = np.zeros((n_chn, n_row, n_col), dtype=np.float)
            measurement_arr = np.zeros_like(score_arr, dtype=np.int)
            model_arr = np.zeros_like(score_arr, dtype=np.object)

            # start iteration
            for chn_idx in range(n_chn):
                for row_idx in range(n_row):
                    for col_idx in range(n_col):
                        y = dnn_acts[:, chn_idx, row_idx, col_idx]
                        score_tmp = []
                        for meas_idx in range(n_meas):
                            X = response[:, meas_idx][:, None]
                            cv_scores = self.model.cross_val_score(X, y, self.cv)
                            score_tmp.append(np.mean(cv_scores))

                        # find max score
                        max_meas_idx = np.argmax(score_tmp)
                        max_score = score_tmp[max_meas_idx]
                        score_arr[chn_idx, row_idx, col_idx] = max_score
                        measurement_arr[chn_idx, row_idx, col_idx] = max_meas_idx + 1

                        # fit the max-score model
                        X = response[:, max_meas_idx][:, None]
                        y = dnn_acts[:, chn_idx, row_idx, col_idx]
                        model_arr[chn_idx, row_idx, col_idx] = self.model.fit(X, y)
                print(f'Finish-{layer}-{chn_idx+1}/{n_chn}')

            pred_dict[layer] = {
                'score': score_arr,
                'measurement': measurement_arr,
                'model': model_arr
            }
        return pred_dict

    def mva(self, response, activation):
        """
        Use responses of brain or behavior to predict DNN activation
        by multivariate analysis.

        Parameters:
        ----------
        response[array]: responses of brain or behavior
            A 2D array with its shape as (n_sample, n_measurement)
        activation[Activation]: DNN activation

        Return:
        ------
        pred_dict[dict]:
            layer:
                score: prediction scores array
                model: fitted models
        """
        n_samp, n_meas = response.shape  # n_sample x n_measures

        pred_dict = dict()
        for layer in activation.layers:
            # get DNN activation
            dnn_acts = activation.get(layer)
            n_stim, n_chn, n_row, n_col = dnn_acts.shape
            assert n_stim == n_samp, 'n_stim != n_samp'

            # prepare containers
            score_arr = np.zeros((n_chn, n_row, n_col), dtype=np.float)
            model_arr = np.zeros_like(score_arr, dtype=np.object)

            # start iteration
            for chn_idx in range(n_chn):
                for row_idx in range(n_row):
                    for col_idx in range(n_col):
                        # calculate score
                        X = response
                        y = dnn_acts[:, chn_idx, row_idx, col_idx]
                        cv_scores = self.model.cross_val_score(X, y, self.cv)

                        # save to containers
                        score_arr[chn_idx, row_idx, col_idx] = np.mean(cv_scores)
                        model_arr[chn_idx, row_idx, col_idx] = self.model.fit(X, y)
                print(f'Finish-{layer}-{chn_idx + 1}/{n_chn}')

            pred_dict[layer] = {
                'score': score_arr,
                'model': model_arr
            }
        return pred_dict
