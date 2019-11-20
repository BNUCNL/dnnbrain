import numpy as np

from dnnbrain.io.fileio import RoiFile
from dnnbrain.dnn.base import UnivariatePredictionModel
from dnnbrain.dnn.base import MultivariatePredictionModel


class ROI:
    """
    A class used to encapsulate and manipulate ROI data of brain
    """
    def __init__(self, fname=None, rois=None):
        """
        Parameters:
        ----------
        fname[str]: file name with suffix as .roi.h5
        rois[str|list]: ROI names of interest
        """
        self.rois = []
        self.data = None
        if fname is None:
            assert rois is None, "The 'rois' only used when 'fname' is not None!"
        else:
            self.load(fname, rois)

    def load(self, fname, rois=None):
        """
        Load from ROI file.

        Parameters:
        ----------
        fname[str]: file name with suffix as .roi.h5
        rois[str|list]: ROI names of interest
        """
        self.rois, self.data = RoiFile(fname).read(rois)

    def save(self, fname):
        """
        Save to ROI file.

        Parameter:
        ---------
        fname[str]: file name with suffix as .roi.h5
        """
        RoiFile(fname).write(self.rois, self.data)

    def get(self, rois):
        """
        Get data according to ROI names

        Parameter:
        ---------
        rois[str|list]: ROI names

        Return:
        ------
        arr[ndarray]: ROI data with shape as (n_vol, n_roi)
        """
        if isinstance(rois, str):
            rois = [rois]

        roi_indices = [self.rois.index(roi) for roi in rois]
        arr = self.data[:, roi_indices]

        return arr

    def set(self, rois, value, index=None):
        """
        Set ROI data with names

        Parameters:
        ----------
        rois[str|list]: ROI names
        value[ndarray]: ROI data
        index[int]: the position where the data is set
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

        Parameter:
        ---------
        rois[str|list]: ROI names
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

        Parameter:
        ---------
        indices[int|list|tuple|slice]: subscript indices

        Return:
        ------
        roi[ROI]: a part of the self.
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

        Parameter:
        ---------
        other[ROI]: ROI object
        """
        if not isinstance(other, ROI):
            raise TypeError("unsupported operand type(s): "
                            "'{0}' and '{1}'".format(type(self), type(other)))
        assert self.rois == other.rois, "The two object's ROIs mismatch!"
        assert self.data.shape == other.data.shape, "The two object's data shape mismatch!"

    def __add__(self, other):
        """
        Do addition between two ROI objects

        Parameter:
        ---------
        other[ROI]: ROI object

        Return:
        ------
        roi[ROI]: ROI object
        """
        self._check_arithmetic(other)

        roi = ROI()
        roi.set(self.rois, self.data + other.data)

        return roi

    def __sub__(self, other):
        """
        Do subtraction between two ROI objects

        Parameter:
        ---------
        other[ROI]: ROI object

        Return:
        ------
        roi[ROI]: ROI object
        """
        self._check_arithmetic(other)

        roi = ROI()
        roi.set(self.rois, self.data - other.data)

        return roi

    def __mul__(self, other):
        """
        Do multiplication between two ROI objects

        Parameter:
        ---------
        other[ROI]: ROI object

        Return:
        ------
        roi[ROI]: ROI object
        """
        self._check_arithmetic(other)

        roi = ROI()
        roi.set(self.rois, self.data * other.data)

        return roi

    def __truediv__(self, other):
        """
        Do true division between two ROI objects

        Parameter:
        ---------
        other[ROI]: ROI object

        Return:
        ------
        roi[ROI]: ROI object
        """
        self._check_arithmetic(other)

        roi = ROI()
        roi.set(self.rois, self.data / other.data)

        return roi


class BrainEncoder:
    """
    Encode DNN activation or behavior data to brain activation.
    """
    def __init__(self, brain_activ=None, model_type=None, model_name=None, cv=3):
        """
        Parameters:
        ----------
        brain_activ[ndarray]: brain activation with shape as (n_vol, n_meas)
            For voxel-wise, n_meas is the number of voxels.
            For ROI-wise, n_meas is the number of ROIs.
        model_type[str]: choices=(uv, mv)
            'uv': univariate prediction model
            'mv': multivariate prediction model
        model_name[str]: name of a model used to do prediction
            If is 'corr', it just uses correlation rather than prediction.
                And the model_type must be 'uv'.
        cv[int]: cross validation fold number
        """
        self.set(brain_activ, model_type, model_name, cv)

    def set(self, brain_activ=None, model_type=None, model_name=None, cv=None):
        """
        Set some attributes

        Parameters:
        ----------
        brain_activ[ndarray]: brain activation with shape as (n_vol, n_meas)
            For voxel-wise, n_meas is the number of voxels.
            For ROI-wise, n_meas is the number of ROIs.
        model_type[str]: choices=(uv, mv)
            'uv': univariate prediction model
            'mv': multivariate prediction model
        model_name[str]: name of a model used to do prediction
            If is 'corr', it just uses correlation rather than prediction.
                And the model_type must be 'uv'.
        cv[int]: cross validation fold number
        """
        if brain_activ is not None:
            self.brain_activ = brain_activ

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

    def encode_dnn(self, dnn_activ, iter_axis=None):
        """
        Encode DNN activation to brain activation.

        Parameters:
        ----------
        dnn_activ[Activation]: DNN activation
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
                    shape=(n_iter, n_meas)
                model[ndarray]: fitted models of the max scores
                    shape=(n_iter, n_meas)
                chn_loc[ndarray]: channel locations of the max scores
                    shape=(n_iter, n_meas)
                row_loc[ndarray]: row locations of the max scores
                    shape=(n_iter, n_meas)
                col_loc[ndarray]: column locations of the max scores
                    shape=(n_iter, n_meas)
            ---for mv---
            layer:
                score[ndarray]: prediction scores
                    shape=(n_iter, n_meas)
                model[ndarray]: fitted models
                    shape=(n_iter, n_meas)
        """
        n_samp, n_meas = self.brain_activ.shape

        pred_dict = dict()
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

            # start encoding
            score_arr = np.zeros((n_iter, n_meas), dtype=np.float)
            model_arr = np.zeros_like(score_arr, dtype=np.object)
            if isinstance(self.model, UnivariatePredictionModel):
                channel_arr = np.zeros_like(score_arr, dtype=np.int)
                row_arr = np.zeros_like(score_arr, dtype=np.int)
                column_arr = np.zeros_like(score_arr, dtype=np.int)
                for iter_idx in range(n_iter):
                    data = self.model.predict(activ[:, iter_idx, :], self.brain_activ)
                    score_arr[iter_idx] = data['score']
                    model_arr[iter_idx] = data['model']
                    for meas_idx, loc in enumerate(data['location']):
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

                        channel_arr[iter_idx, meas_idx] = chn_idx + 1
                        row_arr[iter_idx, meas_idx] = row_idx + 1
                        column_arr[iter_idx, meas_idx] = col_idx + 1
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
                    data = self.model.predict(activ[:, iter_idx, :], self.brain_activ)
                    score_arr[iter_idx] = data['score']
                    model_arr[iter_idx] = data['model']
                    print(f'Layer-{layer} iter-{iter_idx + 1}/{n_iter}')

                pred_dict[layer] = {
                    'score': score_arr,
                    'model': model_arr
                }
        return pred_dict

    def encode_behavior(self, beh_data):
        """
        Encode behavior data to brain activation.

        Parameter:
        ---------
        beh_data[ndarray]: behavior data with shape as (n_stim, n_beh)

        Return:
        ------
        pred_dict[dict]:
            ---for uv---
            score[ndarray]: max scores
                shape=(n_meas,)
            model[ndarray]: fitted models of the max scores
                shape=(n_meas,)
            location[ndarray]: locations of behavior indicators with max scores
                shape=(n_meas,)
            ---for mv---
            score[ndarray]: prediction scores
                shape=(n_meas,)
            model[ndarray]: fitted models
                shape=(n_meas,)
        """
        pred_dict = self.model.predict(beh_data, self.brain_activ)

        return pred_dict
