#import some packages needed
from abc import ABC, abstractmethod


class ImagePixelActivation(ABC):
    """ 
    An Abstract Base Classes class to define interface for pixel activation, 
    which compute the activation for each pixel of an image
    """
    def __init__(self, model, metric = None):
        self.model = model
        self.metric = metric
        
    @abstractmethod
    def set_params(self):
        """ set parames """
        
    @abstractmethod
    def compute(self, image, layer, channel): 
        """The method use _estimate to caculate the pixel activtion"""

class OccluderDiscrepancyMap(ImagePixelActivation):
    def __init__(self, model, metric = None, window = None, stride = None):
        self.model = model
        self.window = window
        self.stride = stride
        self.metric = metric
        
    def set_params(self, window, stride, metric = None):
        """Set necessary parameters for the estimator"""
        self.window = window
        self.stride = stride
        self.metric = metric
    
    def compute(image, layer, channel):
        """ The method do real computation for discrepancy map based on sliding occluder"""
        pass

class UpsamplingActivationMap(ImagePixelActivation):
    def __init__(self, model, metric=None):
        self.model = model
        self.metric = metric
    
    def set_params(self):
        """Set necessary parameters for the estimator"""
        pass
    
    def compute(image, layer, channel):
        """ The method do real computation for pixel activation based on feature mapping upsampling"""
        pass


