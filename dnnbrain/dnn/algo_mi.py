#import some packages needed
import numpy as np
import math
import torch
from torch.nn import ReLU
from torch.optim import Adam
from abc import ABC, abstractmethod

class Algorithm(ABC):
    """ 
    An Abstract Base Classes class to define interface for dnn algorithm 
    """
    def __init__(self, dnn, layer = None, channel = None):
        self.dnn = dnn
        self.layer = layer
        self.channel = channel
        
    def set_layer(self, layer, channel):
        self.layer = layer
        self.channel = channel
        
    @abstractmethod
    def set_params(self):
        """ set parames """
        
    @abstractmethod
    def compute(self, image): 
        """Please implement your algorithm here"""
        
        
class MinmalParcelImage(Algorithm):
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """
    
    def set_params(self,  decomposer, criterion):
        """Set parameter for the estimator"""
        self.decomposer = decomposer
        self.criterion = criterion

    def compute(self, image):
        """Generate minmal image for image listed in stim object """
        pass 

        
class MinmalComponentImage(Algorithm):
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """
    
    def set_params(self,  decomposer, criterion):
        """Set parameter for the estimator"""
        self.decomposer = decomposer
        self.criterion = criterion

    def compute(self, image):
        """Generate minmal image for image listed in stim object """
        pass