import numpy as np

from dnnbrain.io.fileio import RoiFile
from dnnbrain.dnn.base import UnivariateMapping, MultivariateMapping


class ROI:
    """
    A class used to encapsulate and manipulate ROI data of brain
    """
    def __init__(self, rois=None, value=None):
        """
        Parameters
        ----------
        rois : str, list 
            ROI names of interest.
        value : ndarray 
            ROI data.
        """
        self.rois = []
        self.data = None
        if rois is not None:
            assert value is not None, "The value can't be None when rois is not None!"
            self.set(rois, value)

    def load(self, fname, rois=None):
        """
        Load from ROI file.

        Parameters
        ----------
        fname : str
            File name with suffix as .roi.h5 
        rois : str, list 
            ROI names of interest.
        """
        self.rois, self.data = RoiFile(fname).read(rois)

    def save(self, fname):
        """
        Save to ROI file.

        Parameters
        ----------
        fname : str
            File name with suffix as .roi.h5
        """
        RoiFile(fname).write(self.rois, self.data)

    def get(self, rois):
        """
        Get data according to ROI names

        Parameters
        ----------
        rois : str, list 
            ROI names.

        Return
        ------
        arr : ndarray 
            ROI data with shape as (n_vol, n_roi).
        """
        if isinstance(rois, str):
            rois = [rois]

        roi_indices = [self.rois.index(roi) for roi in rois]
        arr = self.data[:, roi_indices]

        return arr

    def set(self, rois, value, index=None):
        """
        Set ROI data with names

        Parameters
        ----------
        rois : str, list 
            ROI names.
        value : ndarray 
            ROI data.
        index : int 
            The position where the data is set.
        """
        # preprocessing
        if isinstance(rois, str):
            rois = [rois]
        if value.ndim == 1:
            assert len(rois) == 1, "The number of rois mismatches the value's shape."
            value = value.reshape((-1, 1))
        elif value.ndim == 2:
            assert value.shape[1] == len(rois), 'The number of rois mismatches ' \
                                                'the number of columns of the value.'
        else:
            raise ValueError('The number of dimensions of the value must be 1 or 2.')

        if index is None:
            index = len(self.rois)

        # start inserting
        for idx, roi in enumerate(rois):
            self.rois.insert(index+idx, roi)
        if self.data is None:
            self.data = np.zeros((value.shape[0], 0))
        for idx in range(value.shape[1]):
            self.data = np.insert(self.data, index+idx, value[:, idx], 1)

    def delete(self, rois):
        """
        Delete data according to ROI names

        Parameters
        ----------
        rois : str, list 
            ROI names.
        """
        # prepare indices
        if isinstance(rois, str):
            rois = [rois]
        roi_indices = [self.rois.index(roi) for roi in rois]

        # start deleting
        for roi in rois:
            self.rois.remove(roi)

        if len(self.rois) == 0:
            self.data = None
        else:
            self.data = np.delete(self.data, roi_indices, 1)

    def __getitem__(self, indices):
        """
        Get part of the ROI object by imitating 2D array's subscript index

        Parameters
        ----------
        indices : int, list, tuple, slice
            Subscript indices

        Return
        ------
        roi : ROI 
            A part of the self.
        """
        # parse subscript indices
        if isinstance(indices, int):
            # regard it as row index
            # get all columns
            rows = [indices]
            cols = self.rois
        elif isinstance(indices, (slice, list)):
            # regard it all as row indices
            # get all columns
            rows = indices
            cols = self.rois
        elif isinstance(indices, tuple):
            if len(indices) == 0:
                # get all rows and columns
                rows = slice(None, None, None)
                cols = self.rois
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
                cols = self.rois
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
                    cols = [self.rois[cols]]
                elif isinstance(cols, str):
                    # get a column according to an string
                    cols = [cols]
                elif isinstance(cols, list):
                    if np.all([isinstance(i, int) for i in cols]):
                        # get columns according to a list of integers
                        cols = [self.rois[i] for i in cols]
                    elif np.all([isinstance(i, str) for i in cols]):
                        # get columns according to a list of strings
                        pass
                    else:
                        raise IndexError("only integer [list], string [list] and slices (`:`) "
                                         "are valid column indices")
                elif isinstance(cols, slice):
                    # get columns according to a slice
                    cols = self.rois[cols]
                else:
                    raise IndexError("only integer [list], string [list] and slices (`:`) "
                                     "are valid column indices")
            else:
                raise IndexError("This is a 2D data, "
                                 "and can't support more than 3 subscript indices!")
        else:
            raise IndexError("only integer, slices (`:`), list and tuple are valid indices")

        # get part of self
        roi = ROI()
        roi.set(cols, self.get(cols)[rows])

        return roi

    def _check_arithmetic(self, other):
        """
        Check availability of the arithmetic operation for self

        Parameters
        ----------
        other : ROI 
            ROI object.
        """
        if not isinstance(other, ROI):
            raise TypeError("unsupported operand type(s): "
                            "'{0}' and '{1}'".format(type(self), type(other)))
        assert self.rois == other.rois, "The two object's ROIs mismatch!"
        assert self.data.shape == other.data.shape, "The two object's data shape mismatch!"

    def __add__(self, other):
        """
        Do addition between two ROI objects

        Parameters
        ----------
        other : ROI 
            ROI object.

        Return
        ------
        roi : ROI 
            ROI object.
        """
        self._check_arithmetic(other)

        roi = ROI()
        roi.set(self.rois, self.data + other.data)

        return roi

    def __sub__(self, other):
        """
        Do subtraction between two ROI objects.

        Parameters
        ----------
        other : ROI 
            ROI object.

        Return
        ------
        roi : ROI 
            ROI object.
        """
        self._check_arithmetic(other)

        roi = ROI()
        roi.set(self.rois, self.data - other.data)

        return roi

    def __mul__(self, other):
        """
        Do multiplication between two ROI objects.

        Parameters
        ----------
        other : ROI 
            ROI object.

        Return
        ------
        roi : ROI 
            ROI object.
        """
        self._check_arithmetic(other)

        roi = ROI()
        roi.set(self.rois, self.data * other.data)

        return roi

    def __truediv__(self, other):
        """
        Do true division between two ROI objects

        Parameters
        ----------
        other : ROI 
            ROI object.

        Return
        ------
        roi : ROI 
            ROI object.
        """
        self._check_arithmetic(other)

        roi = ROI()
        roi.set(self.rois, self.data / other.data)

        return roi


class BrainEncoder:
    """
    Encode DNN activation or behavior data to brain activation.
    """
    def __init__(self, brain_activ=None, map_type=None, estimator=None,
                 cv=5, scoring=None):
        """
        Parameters
        ----------
        brain_activ : ndarray 
            Brain activation with shape as (n_vol, n_meas).
            For voxel-wise, n_meas is the number of voxels.
            For ROI-wise, n_meas is the number of ROIs.
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
        self.set_activ(brain_activ)
        self.set_mapper(map_type, estimator, cv, scoring)

    def set_activ(self, brain_activ):
        """
        Set brain activation

        Parameters
        ----------
        brain_activ : ndarray 
            Brain activation with shape as (n_vol, n_meas).
            For voxel-wise, n_meas is the number of voxels.
            For ROI-wise, n_meas is the number of ROIs.
        """
        self.brain_activ = brain_activ

    def set_mapper(self, map_type, estimator, cv, scoring):
        """
        Set UnivariateMapping or MultivariateMapping

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
            NOTE: The estimator type can only be regressor or correlation
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

        if self.mapper.estimator_type not in ('regressor', 'correlation'):
            raise ValueError("Not supported estimator type: {}".format(self.mapper.estimator_type))

    def encode_dnn(self, dnn_activ, iter_axis=None):
        """
        Encode DNN activation to brain activation.

        Parameters
        ----------
        dnn_activ : Activation
            DNN activation.
        iter_axis : None or str
            Iterate along the specified axis. Different map types have different operation.
            
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

        Return
        ------
        encode_dict : dict
            It depends on map type.
           
            +-------+---------+-----------------------------------------------------------------------------+
            |       |         |                           First value                                       |
            |       |         +-----------+-----------------------------------------------------------------+
            | Map   |First    |Second     |                       Second value                              |
            | type  |key      |key        |                                                                 |
            +=======+=========+===========+=================================================================+
            |  uv   | layer   | score     |If estimator type is correlation, it's an array with shape |br|  |
            |       |         |           |as (n_iter, n_meas). |br|                                        |
            |       |         |           |Each element is the maximal pearson r among all features at |br| |
            |       |         |           |corresponding iteration correlating to the corresponding |br|    |
            |       |         |           |measurement. |br|                                                |
            |       |         |           |If estimator type is regressor, it's an array with shape as |br| |
            |       |         |           |(n_iter, n_meas, cv). |br|                                       |
            |       |         |           |For each iteration and measurement, the third axis contains |br| |
            |       |         |           |scores of each cross validation folds, when using the |br|       |
            |       |         |           |feature with maximal score to predict the corresponding |br|     |
            |       |         |           |measurement.                                                     |
            |       | (str)   +-----------+-----------------------------------------------------------------+
            |       |         | location  |An array with shape as (n_iter, n_meas, 3) |br|                  |
            |       |         |           |Max locations of the max scores, the size 3 of the third |br|    |
            |       |         |           |dimension means channel, row and column respectively.            |
            |       |         +-----------+-----------------------------------------------------------------+
            |       |         | model     |An array with shape as (n_iter, n_meas). |br|                    |
            |       |         |           |Fitted models of the max scores. |br|                            |
            |       |         |           |**Note**: only exists when estimator type is regressor           |
            +-------+---------+-----------+-----------------------------------------------------------------+
            |  mv   | layer   | score     |A array with shape as (n_iter, n_meas, cv). |br|                 |
            |       |         |           |The third dimension means scores of each cross validation  |br|  |
            |       | (str)   |           |folds at each iteration and measurement.                         |
            |       |         +-----------+-----------------------------------------------------------------+
            |       |         | model     |A array with shape as (n_iter, n_meas). |br|                     |
            |       |         |           |Each element is a model fitted at the corresponding |br|         |
            |       |         |           |iteration and measurement.                                       |
            +-------+---------+-----------+-----------------------------------------------------------------+

            .. |br| raw:: html

               <br/>
        """
        _, n_meas = self.brain_activ.shape

        encode_dict = dict()
        for layer in dnn_activ.layers:
            # get DNN activation and reshape it to 3D
            activ = dnn_activ.get(layer)
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
                encode_dict[layer] = {'score': np.zeros((n_iter, n_meas))}
            else:
                encode_dict[layer] = {
                    'score': np.zeros((n_iter, n_meas, self.mapper.cv)),
                    'model': np.zeros((n_iter, n_meas), dtype=np.object),
                }

            # start encoding
            if isinstance(self.mapper, UnivariateMapping):
                encode_dict[layer]['location'] = np.zeros((n_iter, n_meas, 3), dtype=np.int)

                # start iteration
                for iter_idx in range(n_iter):
                    data = self.mapper.map(activ[:, iter_idx, :], self.brain_activ)
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
                            encode_dict[layer][k][iter_idx, :, 0] = chn_idx + 1
                            encode_dict[layer][k][iter_idx, :, 1] = row_idx + 1
                            encode_dict[layer][k][iter_idx, :, 2] = col_idx + 1
                        else:
                            encode_dict[layer][k][iter_idx] = v
                    print('Layer-{} iter-{}/{}'.format(layer, iter_idx+1, n_iter))
            else:
                # start iteration
                for iter_idx in range(n_iter):
                    data = self.mapper.map(activ[:, iter_idx, :], self.brain_activ)
                    for k, v in data.items():
                        encode_dict[layer][k][iter_idx] = v
                    print('Layer-{} iter-{}/{}'.format(layer, iter_idx+1, n_iter))

        return encode_dict

    def encode_behavior(self, beh_data):
        """
        Encode behavior data to brain activation.

        Parameters
        ----------
        beh_data : ndarray 
            Behavior data with shape as (n_stim, n_beh).

        Returns
        -------
        encode_dict : dict
            It depends on map type.
            
            +-------+---------+-----------------------------------------------------------------------------+
            |       |         |                           First value                                       |
            |       |         +-----------+-----------------------------------------------------------------+
            | Map   |First    |Second     |                       Second value                              |
            | type  |key      |key        |                                                                 |
            +=======+=========+===========+=================================================================+
            |  uv   | layer   | score     |If estimator type is correlation, it's an array with shape |br|  |
            |       |         |           |as (n_meas,) of max scores. |br|                                 |
            |       |         |           |If estimator type is regressor, it's an array with shape as |br| |
            |       |         |           |(n_meas, cv). |br|                                               |
            |       |         |           |The second dimension contains scores of each |br|                |
            |       |         |           |cross validation fold at maximal location.                       |
            |       | (str)   +-----------+-----------------------------------------------------------------+
            |       |         | location  |An array with shape as (n_meas,). |br|                           |
            |       |         |           |Max locations of the max scores.                                 |
            |       |         +-----------+-----------------------------------------------------------------+
            |       |         | model     |An array with shape as (n_meas,). |br|                           |
            |       |         |           |Fitted models of the max scores. |br|                            |
            |       |         |           |Note: only exists when model is regressor.                       |
            +-------+---------+-----------+-----------------------------------------------------------------+
            |  mv   | layer   | score     |An array with shape as (n_meas, cv). |br|                        |
            |       |         |           |The second dimension contains scores of each cross |br|          |
            |       | (str)   |           |validation  fold.                                                |
            |       |         +-----------+-----------------------------------------------------------------+
            |       |         | model     |An array with shape as (n_meas,). |br|                           |
            |       |         |           |Each element is a model fitted at the corresponding |br|         |
            |       |         |           |measurement.                                                     |
            +-------+---------+-----------+-----------------------------------------------------------------+

            .. |br| raw:: html

               <br/>
        """
        encode_dict = self.mapper.map(beh_data, self.brain_activ)

        return encode_dict


class BrainDecoder:
    """
    Decode brain activation to DNN activation or behavior data.
    """
    def __init__(self, brain_activ=None, map_type=None, estimator=None,
                 cv=5, scoring=None):
        """
        Parameters
        ----------
        brain_activ : ndarray 
            Brain activation with shape as (n_vol, n_meas).
            For voxel-wise, n_meas is the number of voxels.
            For ROI-wise, n_meas is the number of ROIs.
        map_type : str
            choices=(uv, mv).
            uv: univariate mapping.
            mv: multivariate mapping.
        estimator : str | sklearn estimator or pipeline
            If is str, it is a name of a estimator used to do mapping.
            If is 'corr', it just uses correlation rather than prediction.
                And the map_type must be 'uv'.
        cv : int
            the number of cross validation folds.
        scoring : str or callable
            the method to evaluate the predictions on the test set.
        """
        self.set_activ(brain_activ)
        self.set_mapper(map_type, estimator, cv, scoring)

    def set_activ(self, brain_activ):
        """
        Set brain activation

        Parameters
        ----------
        brain_activ : ndarray
            Brain activation with shape as (n_vol, n_meas).
            For voxel-wise, n_meas is the number of voxels.
            For ROI-wise, n_meas is the number of ROIs.
        """
        self.brain_activ = brain_activ

    def set_mapper(self, map_type, estimator, cv, scoring):
        """
        Set mapper parameters.

        Parameters
        ----------
        map_type : str
            choices=(uv, mv).
            uv: univariate mapping.
            mv: multivariate mapping.
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

    def decode_dnn(self, dnn_activ):
        """
        Decode brain activation to DNN activation.

        Parameters
        ---------
        dnn_activ : Activation  
            DNN activation.

        Returns
        -------
        decode_dict : dict
            It depends on map type.
                
            +-------+---------+----------------------------------------------------------------------------+
            |       |         |                           First value                                      |
            |       |         +-----------+----------------------------------------------------------------+
            | Map   |First    |Second     |                       Second value                             |
            | type  |key      |key        |                                                                |
            +=======+=========+===========+================================================================+
            |  uv   | layer   | score     |If estimator type is correlation, it's an array with shape |br| |
            |       |         |           |as (n_chn, n_row, n_col) of max scores. |br|                    |
            |       |         |           |If estimator type is regressor, it's an array with shape |br|   |
            |       |         |           |as (n_chn, n_row, n_col, cv). |br|                              |
            |       |         |           |The forth dimension contains scores of each cross |br|          |
            |       |         |           |validation fold of the max scores.                              |
            |       | (str)   +-----------+----------------------------------------------------------------+
            |       |         | location  |An array with shape as (n_chn, n_row, n_col). |br|              |
            |       |         |           |Locations of measurement indicators with max scores. |br|       |
            |       |         +-----------+----------------------------------------------------------------+
            |       |         | model     |An array with shape as (n_chn, n_row, n_col). |br|              |
            |       |         |           |fitted models of the max scores. |br|                           |
            |       |         |           |**Note**: only exists when model is regressor                   |
            +-------+---------+-----------+----------------------------------------------------------------+
            |  mv   | layer   | score     |An array with shape as (n_chn, n_row, n_col, cv). |br|          |
            |       |         |           |The forth dimension contains scores of each |br|                |
            |       | (str)   |           |cross validation fold at each unit.                             |
            |       |         +-----------+----------------------------------------------------------------+
            |       |         | model     |An array with shape as (n_chn, n_row, n_col). |br|              |
            |       |         |           |Each element is a model fitted at the corresponding unit.       |
            +-------+---------+-----------+----------------------------------------------------------------+

            .. |br| raw:: html

               <br/>
        """
        if self.mapper.estimator_type not in ('regressor', 'correlation'):
            raise ValueError("Not supported estimator type: {}".format(self.mapper.estimator_type))

        decode_dict = dict()
        for layer in dnn_activ.layers:
            # get DNN activation
            activ = dnn_activ.get(layer)
            n_stim, *shape = activ.shape
            activ = activ.reshape((n_stim, -1))

            data = self.mapper.map(self.brain_activ, activ)
            for k, v in data.items():
                if k == 'score':
                    if self.mapper.estimator_type == 'correlation':
                        data[k] = v.reshape(shape)
                    else:
                        data[k] = v.reshape(shape+[self.mapper.cv])
                else:
                    data[k] = v.reshape(shape)
            decode_dict[layer] = data

            print('Layer-{} finished.'.format(layer))

        return decode_dict

    def decode_behavior(self, beh_data):
        """
        Decode brain activation to behavior data.

        Parameters
        ----------
        beh_data : ndarray 
            Behavior data with shape as (n_stim, n_beh).

        Returns
        -------
        decode_dict : dict
            It depends on map type.

            +----------+-----------+--------------------------------------------------------------------+
            | map type | key       | value                                                              |
            +==========+===========+====================================================================+
            | uv       | score     | If estimator type is correlation, it's an array with shape as |br| |
            |          |           | (n_beh,). |br|                                                     |
            |          |           | Each element is the maximal pearson r among all measurements |br|  |
            |          |           | correlating to the corresponding behavior. |br|                    |
            |          |           | If estimator type is regressor or classifier, it's an array |br|   |
            |          |           | with shape as (n_beh, cv). |br|                                    |
            |          |           | Each row contains scores of each cross validation fold, when |br|  |
            |          |           | using the measurement at the maximal location to predict the |br|  |
            |          |           | corresponding behavior.                                            |
            |          +-----------+--------------------------------------------------------------------+
            |          | location  | An array with shape as (n_beh,). |br|                              |
            |          |           | Each element is a location of the measurement which makes the |br| |
            |          |           | maximal score.                                                     |
            |          +-----------+--------------------------------------------------------------------+
            |          | model     | An array with shape as (n_beh,). |br|                              |
            |          |           | Each element is a model fitted by the measurement at the |br|      |
            |          |           | maximal location and the corresponding behavior. |br|              |
            |          |           | **Note**: not exist when estimator type is correlation             |
            |          +-----------+--------------------------------------------------------------------+
            |          | conf_m    | An array with shape as (n_beh, cv). |br|                           |
            |          |           | Each row contains confusion matrices (n_label, n_label) of |br|    |
            |          |           | each cross validation fold, when using the measurement at the |br| |
            |          |           | maximal location to predict the corresponding behavior. |br|       |
            |          |           | **Note**: only exists when estimator type is classifier            |
            +----------+-----------+--------------------------------------------------------------------+
            |  mv      | score     | An array with shape as (n_beh, cv). |br|                           |
            |          |           | Each row contains scores of each cross validation fold, when |br|  |
            |          |           | using all measurements to predict the corresponding behavior.      |
            |          +-----------+--------------------------------------------------------------------+
            |          | model     | An array with shape as (n_beh,). |br|                              |
            |          |           | Each element is a model fitted by all measurements and the |br|    |
            |          |           | corresponding behavior.                                            |
            |          +-----------+--------------------------------------------------------------------+
            |          | conf_m    | An array with shape as (n_beh, cv). |br|                           |
            |          |           | Each row contains confusion matrices (n_label, n_label) of |br|    |
            |          |           | each cross validation fold, when using all measurements to |br|    |
            |          |           | predict the corresponding behavior. |br|                           |
            |          |           | **Note**: only exists when estimator type is classifier            |
            +----------+-----------+--------------------------------------------------------------------+

            .. |br| raw:: html

               <br/>
        """
        decode_dict = self.mapper.map(self.brain_activ, beh_data)

        return decode_dict
