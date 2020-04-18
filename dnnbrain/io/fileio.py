import h5py
import numpy as np

from collections import OrderedDict


class StimulusFile:
    """A class to read and write stimulus file """

    def __init__(self, fname):
        """
        Parameter:
        ---------
        fname[str]: pre-designed .stim.csv file.
        Format of .stim.csv of image stimuli is
        --------------------------
        type=image
        path=parent_dir_to_images
        [Several optional keys] (eg., title=image stimuli)
        data=stimID,[onset],[duration],[label],[condition],acc,RT
        pic1_name,0,1,0,cat,0.4,0.5
        pic2_name,1,1,1,dog,0.6,0.4
        pic3_name,2,1,0,cat,0.7,0.5
        ...,...,...,...,...,...

        Format of .stim.csv of video stimuli is
        --------------------------
        type=video
        path=path_of_video_file
        [Several optional keys] (eg., title=video stimuli)
        data=stimID,[onset],[duration],[label],[condition],acc,RT
        1,0,1,0,cat,0.4,0.5
        2,1,1,1,dog,0.6,0.4
        3,2,1,0,cat,0.7,0.5
        ...,...,...,...,...,...
        """
        assert fname.endswith('.stim.csv'), "File suffix must be .stim.csv"
        self.fname = fname
        
    def read(self):
        """
        Return:
        -------
        stimuli[OrderedDict]: Dictionary of the stimuli information
        """
        # -load csv data-
        with open(self.fname) as rf:
            lines = rf.read().splitlines()
        # remove null line
        while '' in lines:
            lines.remove('')
        data_idx = [line.startswith('data=') for line in lines].index(True)
        meta_lines = lines[:data_idx]
        var_lines = lines[data_idx+1:]

        # -handle csv data-
        # --operate meta_lines--
        stimuli = {}
        for line in meta_lines:
            k, v = line.split('=')
            stimuli[k] = v
        assert 'type' in stimuli.keys(), "'type' needs to be included in meta data."
        assert 'path' in stimuli.keys(), "'path' needs to be included in meta data."
        assert stimuli['type'] in ('image', 'video'), 'not supported type: {}'.format(stimuli['type'])

        # --operate var_lines--
        # prepare keys
        data_keys = lines[data_idx].split('=')[1].split(',')
        assert 'stimID' in data_keys, "'stimID' must be included in 'data=' line."

        # prepare variable data
        var_data = [line.split(',') for line in var_lines]
        var_data = list(zip(*var_data))

        # fill variable data
        data = OrderedDict()
        for idx, key in enumerate(data_keys):
            if key == 'stimID':
                if stimuli['type'] == 'image':
                    dtype = np.str
                else:
                    var_data[idx] = np.float64(var_data[idx])
                    dtype = np.int
            elif key == 'label':
                var_data[idx] = np.float64(var_data[idx])
                dtype = np.int
            elif key == 'condition':
                dtype = np.str
            else:
                dtype = np.float
            data[key] = np.asarray(var_data[idx], dtype=dtype)
        stimuli['data'] = data

        return stimuli
    
    def write(self, type, path, data, **opt_meta):
        """
        Parameters:
        ----------
        type[str]: stimulus type in ('image', 'video')
        path[str]: path_to_stimuli
            If type is 'image', the path is the parent directory of the images.
            If type is 'video', the path is the file name of the video.
        data[dict]: stimulus variable data
        opt_meta[dict]: some other optional meta data
        """
        with open(self.fname, 'w') as wf:
            # write the type and path
            wf.write('type={}\n'.format(type))
            wf.write('path={}\n'.format(path))

            # write optional meta data
            for k, v in opt_meta.items():
                wf.write('{0}={1}\n'.format(k, v))

            # write variable data
            wf.write('data={}\n'.format(','.join(data.keys())))
            var_data = np.array(list(data.values())).astype(np.str).T
            var_data = [','.join(row) for row in var_data]
            wf.write('\n'.join(var_data))


class ActivationFile:
    """a class to read and write activation file """

    def __init__(self, fname):
        """
        Parameter:
        ---------
        fname[str]: file name with suffix as .act.h5
        """
        assert fname.endswith('.act.h5'), "the file's suffix must be .act.h5"
        self.fname = fname

    def read(self, dmask=None):
        """
        Read DNN activation

        Parameter:
        ---------
        dmask[dict]: Dictionary of the DNN mask information

        Return:
        ------
        activation[dict]: DNN activation
        """
        # prepare
        rf = h5py.File(self.fname, 'r')

        if dmask is None:
            dmask = dict()
            for layer in rf.keys():
                dmask[layer] = {'chn': 'all', 'row': 'all', 'col': 'all'}

        # read activation
        activation = dict()
        for k, v in dmask.items():
            activation[k] = dict()
            ds = rf[k]
            if v['chn'] != 'all':
                channels = [chn-1 for chn in v['chn']]
                ds = ds[:, channels, :, :]
            if v['row'] != 'all':
                rows = [row-1 for row in v['row']]
                ds = ds[:, :, rows, :]
            if v['col'] != 'all':
                columns = [col-1 for col in v['col']]
                ds = ds[:, :, :, columns]

            activation[k] = np.asarray(ds)

        rf.close()
        return activation
    
    def write(self, activation):
        """
        Write DNN activation to a hdf5 file

        Parameter:
        ---------
        activation[dict]: DNN activation
        """
        wf = h5py.File(self.fname, 'w')
        for layer, data in activation.items():
            wf.create_dataset(layer, data=data, compression='gzip')

        wf.close()


class MaskFile:
    """a class to read and write dnn mask file"""

    def __init__(self, fname):
        """
        Parameter:
        ---------
        fname[str]: pre-designed .dmask.csv file
        """
        assert fname.endswith('.dmask.csv'), 'File suffix must be .dmask.csv'
        self.fname = fname
        
    def read(self):
        """ 
        Read DNN mask
    
        Return:
        ------
        dmask[OrderedDict]: Dictionary of the DNN mask information
        """
        # -load csv data-
        with open(self.fname) as rf:
            lines = rf.read().splitlines()

        # extract layers, channels, rows, and columns of interest
        dmask = OrderedDict()
        for l_idx, line in enumerate(lines):
            if '=' in line:
                # layer
                layer, axes = line.split('=')
                dmask[layer] = {'chn': 'all', 'row': 'all', 'col': 'all'}

                # channels, rows, and columns
                axes = axes.split(',')
                while '' in axes:
                    axes.remove('')
                assert len(axes) <= 3, \
                    "The number of a layer's axes must be less than or equal to 3."
                for a_idx, axis in enumerate(axes, 1):
                    assert axis in ('chn', 'row', 'col'), \
                        'Axis must be from (chn, row, col).'
                    numbers = [int(num) for num in lines[l_idx + a_idx].split(',')]
                    dmask[layer][axis] = numbers

        return dmask

    def write(self, dmask):
        """
        Generate .dmask.csv

        Parameters:
        ----------
        dmask[dict]: Dictionary of the DNN mask information
        """
        with open(self.fname, 'w') as wf:
            for layer, axes_dict in dmask.items():
                axes = []
                num_lines = []
                assert len(axes_dict) <= 3, \
                    "The number of a layer's axes must be less than or equal to 3."
                for axis, numbers in axes_dict.items():
                    assert axis in ('chn', 'row', 'col'), \
                        'Axis must be from (chn, row, col).'
                    if numbers != 'all':
                        axes.append(axis)
                        num_line = ','.join(map(str, numbers))
                        num_lines.append(num_line)

                wf.write('{0}={1}\n'.format(layer, ','.join(axes)))
                for num_line in num_lines:
                    wf.write(num_line + '\n')


class RoiFile:
    """a class to read and write roi file """

    def __init__(self, fname):
        """
        Parameter:
        ---------
        fname[str]: pre-designed .roi.h5 file
        """
        assert fname.endswith('.roi.h5'), "the file's suffix must be .roi.h5"
        self.fname = fname
            
    def read(self, rois=None):
        """
        Read data of ROIs of the brain.

        Parameter:
        ---------
        rois[str|list]: ROI names of interest

        Return:
        ------
        rois[list]: ROI names which are corresponding to columns in data
        data[ndarray]: ROI data
        """
        rf = h5py.File(self.fname, 'r')
        if rois is None:
            rois = rf.attrs['roi'].tolist()
            data = rf['data'][:]
        else:
            if isinstance(rois, str):
                rois = [rois]
            rois_all = rf.attrs['roi'].tolist()
            roi_indices = [rois_all.index(roi) for roi in rois]
            data = rf['data'][:, roi_indices]

        rf.close()
        return rois, data

    def write(self, rois, data):
        """
        Write data of ROIs of the brain.

        Parameters:
        ---------
        rois[str|list]: ROI names which are corresponding to columns in data
        data[ndarray]: ROI data
        """
        # preprocessing
        if isinstance(rois, str):
            rois = [rois]
        if data.ndim == 1:
            assert len(rois) == 1, 'The number of rois mismatches the data shape.'
            data = data.reshape((-1, 1))
        elif data.ndim == 2:
            assert data.shape[1] == len(rois), 'The number of rois mismatches ' \
                                               'the number of columns of the data.'
        else:
            raise ValueError('The number of dimensions of the data must be 1 or 2.')

        # write to file
        wf = h5py.File(self.fname, 'w')
        wf.attrs['roi'] = rois
        wf.create_dataset('data', data=data)
        wf.close()


class RdmFile:
    """
    a class to read/write representation distance matrices from/to .rdm.h5 file
    For saving time and space, DNNBrain only hold on the lower triangle of each RDM.
    We can use np.tri(n_item, k=-1, dtype=np.bool) to get the index matrix of the lower triangle.
    The index matrix can help us to restore RDM from the lower triangle whose shape is ((n_item^2-n_item)/2,).
    """

    def __init__(self, fname):
        """
        Parameter:
        ---------
        fname[str]: file name with suffix as .rdm.h5
        """
        assert fname.endswith('.rdm.h5'), "the file's suffix must be .rdm.h5"
        self.fname = fname

    def read(self):
        """
        Read RDMs from .rdm.h5 file

        Return:
        ------
        rdm_type[str]: choices=(bRDM, dRDM)
            bRDM: RDM for brain activation
            dRDM: RDM for DNN activation
        rdm_dict[dict]:
            If rdm_type is bRDM:
                The keys of rdm_dict are labels of brain ROIs;
                The values of rdm_dict are arrays with shape as ((n_item^2-n_item)/2,)
            If rdm_type is dRDM:
                The keys of rdm_dict are layer names;
                The values of rdm_dict are arrays with shape as (n_iter, (n_item^2-n_item)/2)
        """
        rf = h5py.File(self.fname, 'r')

        # check
        rdm_type = rf.attrs['type']
        if rdm_type == 'bRDM':
            for v in rf.values():
                assert v.ndim == 1, "bRDM's shape must be ((n_item^2-n_item)/2,)"
        elif rdm_type == 'dRDM':
            for v in rf.values():
                assert v.ndim == 2, "dRDM's shape must be (n_iter, (n_item^2-n_item)/2)"
        else:
            raise TypeError("RDM type must be one of (bRDM, dRDM)")

        # get data
        rdm_dict = dict()
        for k, v in rf.items():
            rdm_dict[k] = v[:]

        rf.close()
        return rdm_type, rdm_dict

    def write(self, rdm_type, rdm_dict):
        """
        write RDMs to .rdm.h5 file

        Parameters:
        ----------
        rdm_type[str]: choices=(bRDM, dRDM)
            bRDM: RDM for brain activation
            dRDM: RDM for DNN activation
        rdm_dict[dict]:
            If rdm_type is bRDM:
                The keys of rdm_dict are labels of brain ROIs;
                The values of rdm_dict are arrays with shape as ((n_item^2-n_item)/2,)
            If rdm_type is dRDM:
                The keys of rdm_dict are layer names;
                The values of rdm_dict are arrays with shape as (n_iter, (n_item^2-n_item)/2)
        """
        # check
        if rdm_type == 'bRDM':
            for v in rdm_dict.values():
                assert v.ndim == 1, "bRDM's shape must be ((n_item^2-n_item)/2,)"
        elif rdm_type == 'dRDM':
            for v in rdm_dict.values():
                assert v.ndim == 2, "dRDM's shape must be (n_iter, (n_item^2-n_item)/2)"
        else:
            raise TypeError("RDM type must be one of (bRDM, dRDM)")

        # write
        wf = h5py.File(self.fname, 'w')
        wf.attrs['type'] = rdm_type
        for k, v in rdm_dict.items():
            wf.create_dataset(k, data=v, compression='gzip')
        wf.close()


class ImageFile():
    """a class to read and write image file """
    def __init__(self, file_path):        
        assert file_path.endswith('.png'), "the file's suffix must be .png"
        self.path = file_path
        
    def set(self, file_path):
        self.path = file_path;
        
    def read(self):
        return h5py.File(self.path, 'r')
        
    def write(self, image):
        """
        Write an image object to disk file
        image: a image object
        """
        h5py.File(self.path, image, 'w')
    
class VideoFile():
    """a class to read and write video file """
    def __init__(self, file_path):        
        assert file_path.endswith('.mp4'), "the file's suffix must be .mp4"
        self.path = file_path
        
    def set(self, file_path):
        self.path = file_path;
        
    def read(self):
        return h5py.File(self.path, 'r')
        
    def write(self, video):
        """
        Write an video object to the disk
        video: a video object
        """
        h5py.File(self.path, video, 'w')
