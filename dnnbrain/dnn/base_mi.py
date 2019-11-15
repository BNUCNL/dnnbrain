#import some packages needed
import numpy as np
import math
import torch
from torch.nn import ReLU
from torch.optim import Adam
from abc import ABC
try:
    from misc_functions import preprocess_image, recreate_image
except ModuleNotFoundError:
    pass
    #raise Exception('Please install misc_functions in your work station')
    
class ImageDecomposer(ABC):
    """ 
    An Abstract Base Classes class to define interface for image decomposer, 
    which decompose an image into different parcels or components
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def set_params(self):
        return NotImplemented
    
    @abstractmethod
    def decompose(self, image):
        return NotImplemented
        
class ImageComponentDecomposer(ImageDecomposer):
    """ Use a component model to decompose an image into different components"""
    def __init__(self,component_model):
        self.model = componet
    
    def set_params(self):
        """Set necessary parameters for the decomposer"""
        pass
    
    def decompose(self, image):
        print('Please write code here to decompose image.')
        
class ImageParcelDecomposer(ImageDecomposer):
    """ Use a parcel model to decompose an image into different components"""
    def __init__(self,parcel_model):
        self.model = parcel_model
        
    def set_params(self):
        """Set necessary parameters for the decomposer"""
        pass
        
    def decompose(self, image):
        print('Please write code here to segment image into different parcels.')


class MinmalImageEstimator():
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """
    def __init__(self, dnn = None, part_decomposer=None, optimization_criterion = None):
        self.model = dnn
        self.decomposer = part_decomposer
        self.optimization_criterion = optimization_criterion
        
    def set(self, model, part_decomposer, optimization_criterion):
        self.model = model
        self.decomposer = part_decomposer
        self.optimization_criterion = optimization_criterion
        
    def estimate(self, image, layer, channel):
        """Generate minmal image of input image for a target layer and channel """
        pass