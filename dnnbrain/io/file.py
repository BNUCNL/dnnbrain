import numpy as np
import torch
import h5py

from collections import OrderedDict


class StimulusFile:
    """A class to read and write stimullus file """

    def __init__(self, path):
        """
        Parameter:
        ---------
        path[str]: pre-designed .stim.csv file.
        Format of .stim.csv of image stimuli is
        --------------------------
        type=image
        path=parent_dir_to_images
        [Several optional keys] (eg., title=image stimuli)
        data=stimID,[onset],[duration],[label],[condition],acc,RT
        pic1_path,0,1,0,cat,0.4,0.5
        pic2_path,1,1,1,dog,0.6,0.4
        pic3_path,2,1,0,cat,0.7,0.5
        ...,...,...,...,...,...

        Format of .stim.csv of video stimuli is
        --------------------------
        type=video
        path=path_to_video_file
        [Several optional keys] (eg., title=video stimuli)
        data=stimID,[onset],[duration],[label],[condition],acc,RT
        1,0,1,0,cat,0.4,0.5
        2,1,1,1,dog,0.6,0.4
        3,2,1,0,cat,0.7,0.5
        ...,...,...,...,...,...
        """
        assert path.endswith('.stim.csv'), "File suffix must be .stim.csv"
        self.path = path
        
    def read(self):
        """
        Return:
        -------
        stimuli[OrderedDict]: Dictionary of the stimuli information
        """
        # -load csv data-
        with open(self.path) as rf:
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
                dtype = np.str if stimuli['type'] == 'image' else np.int
            elif key == 'label':
                dtype = np.int
            elif key == 'condition':
                dtype = np.str
            else:
                dtype = np.float
            data[key] = np.array(var_data[idx], dtype=dtype)
        stimuli['data'] = data

        return stimuli
    
    def write(self, type, stim_path, data, **opt_meta):
        """
        Parameters:
        ----------
        type[str]: stimulus type in ('image', 'video')
        stim_path[str]: path_to_stimuli
            If type is 'image', the path is the parent directory of the images.
            If type is 'video', the path is the file path of the video.
        data[dict]: stimulus variable data
        opt_meta[dict]: some other optional meta data
        """
        with open(self.path, 'w') as wf:
            # write the type and path
            wf.write('type={}\n'.format(type))
            wf.write('path={}\n'.format(stim_path))

            # write optional meta data
            for k, v in opt_meta.items():
                wf.write('{0}={1}\n'.format(k, v))

            # write variable data
            wf.write('data={}\n'.format(','.join(data.keys())))
            var_data = np.array(list(data.values()), dtype=np.str).T
            var_data = [','.join(row) for row in var_data]
            wf.write('\n'.join(var_data))


class ActivationFile:
    """a class to read and write activation file """

    def __init__(self, path):
        """
        Parameter:
        ---------
        path[str]: file path with suffix as .act.h5
        """
        assert path.endswith('.act.h5'), "the file's suffix must be .act.h5"
        self.path = path

    def read(self, dmask=None):
        """
        Read DNN activation and its raw_shape (if exist)

        Parameter:
        ---------
        dmask[dict]: Dictionary of the DNN mask information

        Return:
        ------
        act[dict]: DNN activation with its attribution
        """
        # open file
        rf = h5py.File(self.path, 'r')

        # read activation and attribution
        act = dict()
        layers = rf.keys() if dmask is None else dmask.keys()
        for layer in layers:
            act[layer] = dict()
            ds = rf[layer]
            if dmask['chn'] != 'all':
                channels = [chn-1 for chn in dmask['chn']]
                ds = ds[:, channels, :]
            if dmask['col'] != 'all':
                columns = [col-1 for col in dmask['col']]
                ds = ds[:, :, columns]

            act[layer]['data'] = np.asarray(ds)
            if 'raw_shape' in rf[layer].attrs:
                act[layer]['raw_shape'] = tuple(rf[layer].attrs['raw_shape'])

        rf.close()
        return act
    
    def write(self, act):
        """
        Write DNN activation to a hdf5 file

        Parameter:
        ---------
        act[dict]: DNN activation with its attribution
        """
        wf = h5py.File(self.path, 'w')
        for layer, value in act.items():
            ds = wf.create_dataset(layer, data=value['data'])
            if 'raw_shape' in value:
                ds.attrs['raw_shape'] = value['raw_shape']
        wf.close()


class NetFile:
    """a class to read and write net file"""
    def __init__(self, file_path):
        assert file_path.endswith('.pth'), "the file's suffix must be pth"
        self.path = file_path
        
    def set(self, file_path):
        """file_path: path for target file"""
        self.path = file_path
        
    def read(self):
        model = torch.load(self.path)
        return model
    
    def write(self, net):
        """
        Write a net object to a pth file
        net: a pth object
        """
        pass 


class MaskFile:
    """a class to read and write dnn mask file"""

    def __init__(self, path):
        """
        Parameter:
        ---------
        path[str]: pre-designed .dmask.csv file
        """
        assert path.endswith('.dmask.csv'), 'File suffix must be .dmask.csv'
        self.path = path
        
    def read(self):
        """ 
        Read DNN mask
    
        Return:
        ------
        dmask[OrderedDict]: Dictionary of the DNN mask information
        """
        # -load csv data-
        with open(self.path) as rf:
            lines = rf.read().splitlines()

        # extract layers, channels and columns of interest
        dmask = OrderedDict()
        for l_idx, line in enumerate(lines):
            if '=' in line:
                # layer
                layer, axes = line.split('=')
                dmask[layer] = {'chn': 'all', 'col': 'all'}

                # channels and columns
                axes = axes.split(',')
                while '' in axes:
                    axes.remove('')
                assert len(axes) <= 2, \
                    "The number of a layer's axes must be less than or equal to 2."
                for a_idx, axis in enumerate(axes, 1):
                    assert axis in ('chn', 'col'), 'Axis must be from (chn, col).'
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
        with open(self.path, 'w') as wf:
            for layer, axes_dict in dmask.items():
                axes = []
                num_lines = []
                assert len(axes_dict) <= 2, \
                    "The number of a layer's axes must be less than or equal to 2."
                for axis, numbers in axes_dict.items():
                    assert axis in ('chn', 'col'), 'Axis must be from (chn, col).'
                    if numbers != 'all':
                        axes.append(axis)
                        num_line = ','.join(map(str, numbers))
                        num_lines.append(num_line)

                wf.write('{0}={1}\n'.format(layer, ','.join(axes)))
                for num_line in num_lines:
                    wf.write(num_line + '\n')


class RoiFile():
        """a class to read and write roi file """
        def __init__(self, file_path):        
            assert file_path.endswith('.roi.h5'), "the file's suffix must be .roi.h5"
            self.path = file_path
            
        def set(self, file_path):
            """file_path: path for target file"""
            self.path = file_path
            
        def read(self):
            return h5py.File(self.path, 'r')
            
        def write(self, roi):
            """
            Write an activation object to a hdf5 file
            roi: a roi object
            """
            h5py.File(self.path, roi, 'w')
            
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
      
