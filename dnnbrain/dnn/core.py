import numpy as np

from copy import deepcopy
from dnnbrain.io import fileio as fio
from dnnbrain.dnn.base import dnn_mask, dnn_fe, array_statistic
from dnnbrain.dnn.base import UnivariateMapping, MultivariateMapping
from dnnbrain.brain.algo import convolve_hrf


class Stimulus:
    """
    Store and handle stimulus-related information
    """
    def __init__(self, header=None, data=None):
        """
        Parameters
        ----------
        header : dict
            Meta-information of stimuli
        data : dict
            Stimulus/behavior data.
            Its values are arrays with shape as (n_stim,).
            It must have the key 'stimID'.
        """
        if header is None:
            self.header = dict()
        else:
            assert isinstance(header, dict), "header must be dict"
            self.header = header

        if data is None:
            self._data = dict()
        else:
            n_stim = len(data['stimID'])
            for v in data.values():
                assert isinstance(v, np.ndarray), "data's value must be an array."
                assert v.shape == (n_stim,), "data's value must be an array with shape as (n_stim,)"
            self._data = data

    def load(self, fname):
        """
        Load stimulus-related information

        Parameters
        ----------
        fname : str
            File name with suffix as .stim.csv
        """
        stim_file = fio.StimulusFile(fname)
        stimuli = stim_file.read()
        self._data = stimuli.pop('data')
        self.header = stimuli

    def save(self, fname):
        """
        Save stimulus-related information

        Parameters
        ----------
        fname : str
            File name with suffix as .stim.csv
        """
        stim_file = fio.StimulusFile(fname)
        header = self.header.copy()
        stim_file.write(header.pop('type'), header.pop('path'),
                        self._data, **header)

    def get(self, item):
        """
        Get a column of data according to the item

        Parameters
        ----------
        item : str
            Item name of each column

        Returns
        -------
        col : array
            A column of data
        """
        return self._data[item]

    def set(self, item, value):
        """
        Set a column of data according to the item

        Parameters
        ----------
        item : str 
            Item name of the column
        value : array_like
            An array_like data with shape as (n_stim,)
        """
        self._data[item] = np.asarray(value)

    def delete(self, item):
        """
        Delete a column of data according to item

        Parameters
        ----------
        item : str
            Item name of each column
        """
        self._data.pop(item)

    @property
    def items(self):
        """
        Get list of items

        Returns
        -------
        items : list
            The list of items
        """
        return list(self._data.keys())

    def __len__(self):
        """
        the length of the Stimulus object

        Returns
        -------
        length : int
            The number of stimulus IDs
        """
        return len(self._data['stimID'])

    def __getitem__(self, indices):
        """
        Get part of the Stimulus object by imitating 2D array's subscript index

        Parameters
        ----------
        indices : int,list,tuple,slice
            Subscript indices

        Returns
        -------
        stim : Stimulus
            A part of the self.
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
        stim.header = deepcopy(self.header)
        for item in cols:
            stim.set(item, self.get(item)[rows])

        return stim


class Activation:
    """
    DNN activation
    """
    def __init__(self, layer=None, value=None):
        """
        Parameters
        ----------
        layer : str
            Layer name
        value : array
            4D DNN activation array with shape (n_stim, n_chn, n_row, n_col).
            It will be ignored if layer is None.
        """
        if layer is None:
            self._activation = dict()
        else:
            assert value is not None, "value can't be None if layer is not None."
            self.set(layer, value)

    def load(self, fname, dmask=None):
        """
        Load DNN activation

        Parameters
        ----------
        fname : str
            DNN activation file
        dmask : Mask
            The mask includes layers/channels/rows/columns of interest.
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

        Parameters
        ----------
        fname : str
            Output file of DNN activation
        """
        fio.ActivationFile(fname).write(self._activation)

    def get(self, layer):
        """
        Get DNN activation

        Parameters
        ----------
        layer : str
            Layer name

        Returns
        -------
        act_layer : array
            (n_stim, n_chn, n_row, n_col) array
        """
        return self._activation[layer]

    def set(self, layer, value):
        """
        Set DNN activation

        Parameters
        ----------
        layer : str
            Layer name
        value : array
            4D DNN activation array with shape (n_stim, n_chn, n_row, n_col)
        """
        self._activation[layer] = value

    def delete(self, layer):
        """
        Delete DNN activation

        Parameters
        ----------
        layer : str
            Layer name
        """
        self._activation.pop(layer)

    def concatenate(self, activations):
        """
        Concatenate activations from different batches of stimuli

        Parameters
        ----------
        activations : list
            A list of Activation objects

        Returns
        -------
        activation : Activation
            DNN activation
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
        """
        Get all layers in the Activation

        Returns
        -------
        layers : list
           The list of layers.
        """
        return list(self._activation.keys())

    def mask(self, dmask):
        """
        Mask DNN activation

        Parameters
        ----------
        dmask : Mask
            The mask includes layers/channels/rows/columns of interest.

        Returns
        -------
        activation : Activation
            DNN activation
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

        Parameters
        ----------
        method : str
            Pooling method, choices=(max, mean, median, L1, L2)

        Returns
        -------
        activation : Activation
            DNN activation
        """
        activation = Activation()
        for layer, data in self._activation.items():
            data = array_statistic(data, method, (2, 3), True)
            activation.set(layer, data)

        return activation

    def fe(self, method, n_feat, axis=None):
        """
        Extract features of DNN activation

        Parameters
        ----------
        method : str
            Feature extraction method, choices are as follows:

            +-------------+---------------------------------------------+
            | Method name |              Model description              |
            +=============+=============================================+
            |     pca     | use n_feat principal components as features |
            +-------------+---------------------------------------------+
            |    hist     | use histogram of activation as features     |
            |             | Note: n_feat equal-width bins in the        |
            |             | given range will be used!                   |
            +-------------+---------------------------------------------+
            |     psd     | use power spectral density as features      |
            +-------------+---------------------------------------------+
        n_feat : int, float
            The number of features to extract.
            Note: It can be a float only when the method is pca.
        axis : str
            axis for feature extraction, choices=(chn, row_col)

        Returns
        -------
        activation : Activation
            DNN activation
        """
        activation = Activation()
        for layer, data in self._activation.items():
            data = dnn_fe(data, method, n_feat, axis)
            activation.set(layer, data)

        return activation

    def convolve_hrf(self, onsets, durations, n_vol, tr, ops=100):
        """
        Convolve DNN activation with HRF and align with the timeline of BOLD signal

        Parameters
        ----------
        onsets : array_like
            In sec. size = n_event
        durations : array_like
            In sec. size = n_event
        n_vol : int
            The number of volumes of BOLD signal
        tr : float
            Repeat time in second
        ops : int
            Oversampling number per second

        Returns
        -------
        activation : Activation
            DNN activation
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

        Parameters
        ----------
        other : Activation
            DNN activation
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

        Parameters
        ----------
        other : Activation
            DNN activation

        Returns
        -------
        activation : Activation
            DNN activation
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

        Parameters
        ----------
        other : Activation
            DNN activation

        Returns
        -------
        activation : Activation
            DNN activation
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

        Parameters
        ----------
        other : Activation
            DNN activation

        Returns
        -------
        activation : Activation
            DNN activation
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

        Parameters
        ----------
        other : Activation
            DNN activation

        Returns
        -------
        activation : Activation
            DNN activation
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

        Parameters
        ----------
        indices : int, list, slice
            indices of stimulus axis

        Returns
        -------
        activation : Activation
            DNN activation
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
    """
    DNN mask
    """
    def __init__(self, layer=None, channels='all', rows='all', columns='all'):
        """
        Parameters
        ----------
        layer : str
            Layer name.
            If layer is None, other parameters will be ignored.
        channels : str, list
            Channels of interest.
            If is str, it must be 'all' which means all channels.
            If is list, its elements are serial numbers of channels.
        rows : str, list
            Rows of interest.
            If is str, it must be 'all' which means all rows.
            If is list, its elements are serial numbers of rows.
        columns : str, list
            Columns of interest.
            If is str, it must be 'all' which means all columns.
            If is list, its elements are serial numbers of columns.
        """
        self._dmask = dict()
        if layer is not None:
            self.set(layer, channels=channels, rows=rows, columns=columns)

    def load(self, fname):
        """
        Load DNN mask, the whole mask will be overrode.

        Parameters
        ----------
        fname : str 
            DNN mask file
        """
        self._dmask = fio.MaskFile(fname).read()

    def save(self, fname):
        """
        Save DNN mask

        Parameters
        ----------
        fname : str
            Output file name of DNN mask
        """
        fio.MaskFile(fname).write(self._dmask)

    def get(self, layer):
        """
        Get mask of a layer

        Parameters
        ---------
        layer : str
            Layer name

        Returns
        -------
        mask : dict
            The mask of a specific layer
        """
        return self._dmask[layer]

    def set(self, layer, **kwargs):
        """
        Set DNN mask

        Parameters
        ----------
        layer : str
            Layer name.
            If layer is new, its corresponding mask value will be initialized as 'all'.
        kwargs : dict
            Keyword arguments.
            Only three keywords ('channels', 'rows', 'columns') are valid.            
                    
            +-------------+-------------+----------------------------------------------+
            |   Keywords  |    Option   |                Description                   |
            +=============+=============+==============================================+
            |   channels  |     str     | It must be 'all' which means all channels.   |
            |             +-------------+----------------------------------------------+
            |             |    list     | Its elements are serial numbers of channels. |
            +-------------+-------------+----------------------------------------------+
            |     rows    |     str     | It must be 'all' which means all rows.       |
            |             +-------------+----------------------------------------------+
            |             |    list     | Its elements are serial numbers of rows.     |
            +-------------+-------------+----------------------------------------------+
            |   columns   |     str     | It must be 'all' which means all columns.    |
            |             +-------------+----------------------------------------------+
            |             |    list     | Its elements are serial numbers of columns.  |
            +-------------+-------------+----------------------------------------------+
        """
        # assertion
        for k, v in kwargs.items():
            assert k in ('channels', 'rows', 'columns'), \
                "keyword must be one of ('channels', 'rows', 'columns')"
            assert v == 'all' or isinstance(v, list), \
                f"{k} must be 'all' or list of non-negative integers"

        if layer not in self._dmask:
            self._dmask[layer] = {'chn': 'all', 'row': 'all', 'col': 'all'}
        if 'channels' in kwargs:
            self._dmask[layer]['chn'] = kwargs['channels']
        if 'rows' in kwargs:
            self._dmask[layer]['row'] = kwargs['rows']
        if 'columns' in kwargs:
            self._dmask[layer]['col'] = kwargs['columns']

    def copy(self):
        """
        Make a copy of the DNN mask

        Returns
        -------
        dmask : Mask
            The mask includes layers/channels/rows/columns of interest.
        """
        dmask = Mask()
        dmask._dmask = deepcopy(self._dmask)

        return dmask

    def delete(self, layer):
        """
        Delete a layer

        Parameters
        ---------
        Layer : str
            Layer name
        """
        self._dmask.pop(layer)

    def clear(self):
        """
        Empty the DNN mask
        """
        self._dmask.clear()

    @property
    def layers(self):
        """
        Get all layers in the Mask

        Returns
        -------
        layers : list
           The list of layers.
        """
        return list(self._dmask.keys())


class RDM:
    """
    Representation distance matrix
    """
    def __init__(self):
        self.rdm_type = None
        self._rdm_dict = dict()

    def load(self, fname):
        """
        load RDM
        
        Parameters
        ----------
        fname : str
            File name with suffix as .rdm.h5
        """
        self.rdm_type, self._rdm_dict = fio.RdmFile(fname).read()

    def save(self, fname):
        """
        Save RDM
        
        Parameters
        ----------
        fname : str
            File name with suffix as .rdm.h5
        """
        fio.RdmFile(fname).write(self.rdm_type, self._rdm_dict)

    def get(self, key, triu=False):
        """
        Get RDM according its key.

        Parameters
        ----------
        key : str
            The key of the RDM
        triu : bool
            If True, get RDM as the upper triangle vector.
            If False, get RDM as the square matrix.

        Returns
        -------
        rdm_arr : ndarray
            RDM
            
            If rdm_type is bRDM:
            Its shape is ((n_item^2-n_item)/2,) or (n_item, n_item).
            
            If rdm_type is dRDM:
            Its shape is (n_iter, (n_item^2-n_item)/2) or (n_iter, n_item, n_item).
        """
        rdm_arr = self._rdm_dict[key]
        if not triu:
            idx_arr = np.tri(self.n_item, k=-1, dtype=np.bool).T
            if self.rdm_type == 'bRDM':
                rdm_tmp = np.zeros((self.n_item, self.n_item))
                rdm_tmp[idx_arr] = rdm_arr
            elif self.rdm_type == 'dRDM':
                rdm_tmp = np.zeros((rdm_arr.shape[0], self.n_item, self.n_item))
                rdm_tmp[:, idx_arr] = rdm_arr
            else:
                raise TypeError("Set rdm_type to bRDM or dRDM at first!")
            rdm_arr = rdm_tmp

        return rdm_arr

    def set(self, key, rdm_arr, triu=False):
        """
        Set RDM according its key.

        Parameters
        ----------
        key : str
            The key of the RDM
        rdm_arr : ndarray
            RDM
            
            If rdm_type is bRDM:
            Its shape is ((n_item^2-n_item)/2,) or (n_item, n_item).
            
            If rdm_type is dRDM:
            Its shape is (n_iter, (n_item^2-n_item)/2) or (n_iter, n_item, n_item).
        triu : bool
            If True, RDM will be regarded as the upper triangle vector.
            If False, RDM will be regarded as the square matrix.
        """
        if self.rdm_type == 'bRDM':
            if triu:
                assert rdm_arr.ndim == 1, \
                    "If triu is True, bRDM's shape must be ((n_item^2-n_item)/2,)."
                self._rdm_dict[key] = rdm_arr
            else:
                assert rdm_arr.ndim == 2 and rdm_arr.shape[0] == rdm_arr.shape[1], \
                    "If triu is False, bRDM's shape must be (n_item, n_item)."
                self._rdm_dict[key] = rdm_arr[np.tri(rdm_arr.shape[0], k=-1, dtype=np.bool).T]
        elif self.rdm_type == 'dRDM':
            if triu:
                assert rdm_arr.ndim == 2, \
                    "If triu is True, dRDM's shape must be (n_iter, (n_item^2-n_item)/2)."
                self._rdm_dict[key] = rdm_arr
            else:
                assert rdm_arr.ndim == 3 and rdm_arr.shape[1] == rdm_arr.shape[2], \
                    "If triu is False, dRDM's shape must be (n_iter, n_item, n_item)."
                self._rdm_dict[key] = rdm_arr[:, np.tri(rdm_arr.shape[1], k=-1, dtype=np.bool).T]
        else:
            raise TypeError("Set rdm_type to bRDM or dRDM at first!")

    @property
    def keys(self):
        """
        Get keys of RDM dictionary

        Returns
        -------
        keys : list
            The list of keys
        """
        if self._rdm_dict:
            keys = list(self._rdm_dict.keys())
        else:
            raise ValueError("The RDM dictionary is empty.")

        return keys

    @property
    def n_item(self):
        """
        Get the number of items of RDM

        Returns
        -------
        n_item : int
            The number of items
        """
        k = self.keys[0]
        if self.rdm_type == 'bRDM':
            n = self._rdm_dict[k].shape[0]
        elif self.rdm_type == 'dRDM':
            n = self._rdm_dict[k].shape[1]
        else:
            raise TypeError("Set rdm_type to bRDM or dRDM at first!")
        n_item = int((1 + np.sqrt(1+8*n)) / 2)

        return n_item


class DnnProbe:
    """
    Decode DNN activation to behavior data. As a result,
    probe the ability of DNN activation to predict the behavior.
    """
    def __init__(self, dnn_activ=None, map_type=None, estimator=None,
                 cv=5, scoring=None):
        """
        Parameters
        ----------
        dnn_activ : Activation
            DNN activation
        map_type : str
            choices=(uv, mv)
            uv: univariate mapping
            mv: multivariate mapping
        estimator : str | sklearn estimator or pipeline
            If is str, it is a name of a estimator used to do mapping.
            If is 'corr', it just uses correlation rather than prediction.
                And the map_type must be 'uv'.
        cv : int
            the number of cross validation folds.
        scoring : str or callable
            the method to evaluate the predictions on the test set.
        """
        self.set_activ(dnn_activ)
        self.set_mapper(map_type, estimator, cv, scoring)

    def set_activ(self, dnn_activ):
        """
        Set DNN activation

        Parameters
        ----------
        dnn_activ : Activation
            DNN activation
        """
        self.dnn_activ = dnn_activ

    def set_mapper(self, map_type, estimator, cv, scoring):
        """
        Set mapping attributes

        Parameters
        ----------
        map_type : str
            choices=(uv, mv)
            uv: univariate mapping
            mv: multivariate mapping
        estimator : str | sklearn estimator or pipeline
            If is str, it is a name of a estimator used to do mapping.
            If is 'corr', it just uses correlation rather than prediction.
                And the map_type must be 'uv'.
        cv : int
            the number of cross validation folds.
        scoring : str or callable
            the method to evaluate the predictions on the test set.
        """
        if map_type is None:
            return
        elif map_type == 'uv':
            self.mapper = UnivariateMapping(estimator, cv, scoring)
        elif map_type == 'mv':
            self.mapper = MultivariateMapping(estimator, cv, scoring)
        else:
            raise ValueError('map_type must be one of the (uv, mv).')

    def probe(self, beh_data, iter_axis=None):
        """
        Probe the ability of DNN activation to predict the behavior.

        Parameters
        ----------
        beh_data : ndarray
            Behavior data with shape as (n_stim, n_beh)
        iter_axis : str
            Iterate along the specified axis. Different map type have different operations.
            
            +-------+---------+----------------------------------------------------------+
            | map   |iter_axis|  description                                             |
            | type  |         |                                                          |
            +=======+=========+==========================================================+
            | uv    | channel |Summarize the maximal prediction score for each channel   |
            |       +---------+----------------------------------------------------------+
            |       | row_col |Summarize the maximal prediction score for each position  |
            |       |         |(row_idx, col_idx)                                        |
            |       +---------+----------------------------------------------------------+
            |       | None    |Summarize the maximal prediction score for the whole layer|
            +-------+---------+----------------------------------------------------------+
            |  mv   | channel |Multivariate prediction using all units in each channel   |
            |       +---------+----------------------------------------------------------+
            |       | row_col |Multivariate prediction using all units in each           |
            |       |         |position (row_idx, col_idx)                               |
            |       +---------+----------------------------------------------------------+
            |       | None    |Multivariate prediction using all units in the whole layer|
            +-------+---------+----------------------------------------------------------+

        Returns
        -------
        probe_dict : dict
            A dict containing the score information

            +-------+---------+-----------------------------------------------------------------------+
            |       |         |                           First value                                 |
            |       |         +-----------+-----------------------------------------------------------+
            | Map   |First    |Second     |                       Second value                        |
            | type  |key      |key        |                                                           |
            +=======+=========+===========+===========================================================+
            |  uv   | layer   | score     |If estimator type is correlation, it's an array with shape |
            |       |         |           |as (n_iter, n_beh). Each element is the maximal pearson r  |
            |       |         |           |among all features at corresponding iteration correlating  |
            |       |         |           |to the corresponding behavior.                             |
            |       |         |           |If estimator type is regressor or classifier, it's an array|
            |       |         |           |with shape as (n_iter, n_beh, cv). For each iteration and  |
            |       |         |           |behavior, the third axis contains scores of each cross     |
            |       |         |           |validation fold, when using the feature with maximal score |
            |       |         |           |to predict the corresponding behavior.                     |
            |       | (str)   +-----------+-----------------------------------------------------------+
            |       |         | location  |An array with shape as (n_iter, n_beh, 3)                  |
            |       |         |           |Max locations of the max scores, the                       |
            |       |         |           |size 3 of the third dimension means                        |
            |       |         |           |channel, row and column respectively.                      |
            |       |         +-----------+-----------------------------------------------------------+
            |       |         | model     |An array with shape as (n_iter, n_beh).                    |
            |       |         |           |fitted models of the max scores.                           |
            |       |         |           |Note: not exists when estimator type is correlation        |
            |       |         +-----------+-----------------------------------------------------------+
            |       |         | conf_m    |An array with shape as (n_iter, n_beh, cv).                |
            |       |         |           |The third dimension means confusion matrices               |
            |       |         |           |(n_label, n_label) of each cross validation                |
            |       |         |           |fold of the max scores.                                    |
            |       |         |           |Note: only exists when estimator type is classifier        |
            +-------+---------+-----------+-----------------------------------------------------------+
            |  mv   | layer   | score     |An array with shape as (n_iter, n_beh, cv).                |
            |       |         |           |The third dimension means scores of each                   |
            |       | (str)   |           |cross validation fold at each iteration                    |
            |       |         |           |and behavior                                               |
            |       |         +-----------+-----------------------------------------------------------+
            |       |         | model     |An array with shape as (n_iter, n_beh).                    |
            |       |         |           |Each element is a model fitted at the                      |
            |       |         |           |corresponding iteration and behavior.                      |
            |       |         +-----------+-----------------------------------------------------------+
            |       |         | conf_m    |An array with shape as (n_iter, n_beh, cv).                |
            |       |         |           |The third dimension means confusion matrices               |
            |       |         |           |(n_label, n_label) of each cross validation                |
            |       |         |           |fold at the corresponding iteration and behavior.          |
            |       |         |           |Note: only exists when estimator type is classifier.       |
            +-------+---------+-----------+-----------------------------------------------------------+
        """
        _, n_beh = beh_data.shape

        probe_dict = dict()
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

            # prepare layer dict
            if self.mapper.estimator_type == 'correlation':
                probe_dict[layer] = {'score': np.zeros((n_iter, n_beh))}
            elif self.mapper.estimator_type == 'regressor':
                probe_dict[layer] = {
                    'score': np.zeros((n_iter, n_beh, self.mapper.cv)),
                    'model': np.zeros((n_iter, n_beh), dtype=np.object)
                }
            else:
                probe_dict[layer] = {
                    'score': np.zeros((n_iter, n_beh, self.mapper.cv)),
                    'model': np.zeros((n_iter, n_beh), dtype=np.object),
                    'conf_m': np.zeros((n_iter, n_beh, self.mapper.cv), dtype=np.object)
                }

            # start probing
            if isinstance(self.mapper, UnivariateMapping):
                probe_dict[layer]['location'] = np.zeros((n_iter, n_beh, 3), dtype=np.int)

                # start iteration
                for iter_idx in range(n_iter):
                    data = self.mapper.map(activ[:, iter_idx, :], beh_data)
                    for k, v in data.items():
                        if k == 'location':
                            if iter_axis is None:
                                chn_idx = v // n_row_col
                                row_idx = v % n_row_col // n_col
                                col_idx = v % n_row_col % n_col
                            elif iter_axis == 'channel':
                                chn_idx = iter_idx
                                row_idx = v // n_col
                                col_idx = v % n_col
                            else:
                                chn_idx = v
                                row_idx = iter_idx // n_col
                                col_idx = iter_idx % n_col
                            probe_dict[layer][k][iter_idx, :, 0] = chn_idx + 1
                            probe_dict[layer][k][iter_idx, :, 1] = row_idx + 1
                            probe_dict[layer][k][iter_idx, :, 2] = col_idx + 1
                        else:
                            probe_dict[layer][k][iter_idx] = v
                    print('Layer-{} iter-{}/{}'.format(layer, iter_idx+1, n_iter))
            else:
                # start iteration
                for iter_idx in range(n_iter):
                    data = self.mapper.map(activ[:, iter_idx, :], beh_data)
                    for k, v in data.items():
                        probe_dict[layer][k][iter_idx] = v

                    print('Layer-{} iter-{}/{}'.format(layer, iter_idx+1, n_iter))

        return probe_dict
