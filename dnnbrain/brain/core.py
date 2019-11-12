import numpy as np

from dnnbrain.io.fileio import RoiFile


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
