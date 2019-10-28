import numpy as np
import torch
import h5py

class StimulusFile:
    """A class to read and write stimullus file """
    def __init__(self, file_path):
        assert file_path.endswith('.stim.h5'), "The file's suffix must be .stim.h5"
        self.path = file_path
    
    def set(self, file_path):
        """file_path: path for target file"""
        self.path = file_path
        
    def read(self):
        pass
    
    def write(self, stimulus):
        """
        stimulus: a stimulus object
        """
        pass
    
class ActivationFile:
    """a class to read and write activation file """
    def __init__(self, file_path):        
        assert file_path.endswith('.act.h5'), "the file's suffix must be .act.h5"
        self.path = file_path
    
    def set(self, file_path):
       """file_path: path for target file"""
       self.path = file_path
  
    def read(self):
        return h5py.File(self.path, 'r')
        pass
    
    def write(self, activation):
        """
        Write an activation object to a hdf5 file
        activation: a activation object
        """
        h5py.File(self.path, activation, 'w')
    
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
    def __init__(self, file_path):
        """file_path: path for target dmask file"""
        assert file_path.endswith('.act.h5'), "The file's suffix must be .dmask.csv"
        self.path = file_path
    
    def set(self, file_path):
        """file_path: path for target file"""
        self.path = file_path
        
    def read(self):
        """ 
        Read pre-designed .dmask.csv file.
    
        Parameters:
        ----------
        fpath: path of .dmask.csv file
    
        Return:
        ------
        dmask_dict[OrderedDict]: Dictionary of the DNN mask information
        """
        assert self.path.endswith('.dmask.csv'), 'File suffix must be .dmask.csv'
        with open(self.path) as rf:
            lines = rf.read().splitlines()
    
        # extract layers, channels and columns of interest
        dmask_dict = OrderedDict()
        for l_idx, line in enumerate(lines):
            if '=' in line:
                # layer
                layer, axes = line.split('=')
                dmask_dict[layer] = {'chn': None, 'col': None}
    
                # channels and columns
                axes = axes.split(',')
                while '' in axes:
                    axes.remove('')
                assert len(axes) <= 2, \
                    "The number of a layer's axes must be less than or equal to 2."
                for a_idx, axis in enumerate(axes, 1):
                    assert axis in ('chn', 'col'), 'Axis must be from (chn, col).'
                    numbers = [int(num) for num in lines[l_idx+a_idx].split(',')]
                    dmask_dict[layer][axis] = numbers
    
        return dmask_dict
        
  
    def write(self, dmask):
        """
        Write a dmask object to a dmask.h5 file
        dmask: a dmask object
        """
        assert self.path.endswith('.dmask.csv'), 'File suffix must be .dmask.csv'
        with open(self.path, 'w') as wf:
            for layer, axes_dict in dmask_dict.items():
                axes = []
                num_lines = []
                assert len(axes_dict) <= 2, \
                    "The number of a layer's axes must be less than or equal to 2."
                for axis, numbers in axes_dict.items():
                    assert axis in ('chn', 'col'), 'Axis must be from (chn, col).'
                    if numbers is not None:
                        axes.append(axis)
                        num_line = ','.join(map(str, numbers))
                        num_lines.append(num_line)
    
                wf.write('{0}={1}\n'.format(layer, ','.join(axes)))
                for num_line in num_lines:
                    wf.write(num_line+'\n')

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
      
