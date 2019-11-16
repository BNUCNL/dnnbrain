#import some packages needed
import numpy as np
import math
import torch
from torch.nn import ReLU
from torch.optim import Adam
from abc import ABC, abstractmethod

class ImageDecomposer(ABC):
    """ 
    An Abstract Base Classes class to define interface for image decomposer, 
    which decompose an image into different parcels or components
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def set_params(self):
        """ set params for the decomposer """
        
    @abstractmethod
    def decompose(self, image):
        """ decompose the image into multiple parcel or component"""
        
class ImageComponentDecomposer(ImageDecomposer):
    """ Use a component model to decompose an image into different components"""
    def __init__(self,component_model):
        self.model = component_model
    
    def set_params(self, ncomp):
        """Set necessary parameters for the decomposer"""
        self.ncomp = ncomp
    
    def decompose(self, image):
        print('Please write code here to decompose image.')
        
class ImageParcelDecomposer(ImageDecomposer):
    """ Use a parcel model to decompose an image into different components"""
    def __init__(self,parcel_model):
        self.model = parcel_model
        
    def set_params(self, nparcel):
        """Set necessary parameters for the decomposer"""
        self.nparcel = nparcel
        
    def decompose(self, image):
        print('Please write code here to segment image into different parcels.')