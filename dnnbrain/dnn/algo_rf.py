#import some packages needed
from abc import ABC, abstractmethod


class ImagePixelActivation(ABC):
    """ 
    An Abstract Base Classes class to define interface for pixel activation, 
    which compute the activation for each pixel of an image
    """
    def __init__(self, dnn, metric = None):
        self.dnn = dnn
        self.metric = metric
        
    @abstractmethod
    def set_params(self):
        """ set parames """
        
    @abstractmethod
    def compute(self, image, layer, channel): 
        """The method use _estimate to caculate the pixel activtion"""

class OccluderDiscrepancyMap(ImagePixelActivation):
    def __init__(self, dnn, metric = None, window = None, stride = None):
        self.dnn = dnn
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
    def __init__(self, dnn, metric=None):
        self.dnn = dnn
        self.metric = metric
    
    def set_params(self):
        """Set necessary parameters for the estimator"""
        pass
    
    def compute(image, layer, channel):
        """ The method do real computation for pixel activation based on feature mapping upsampling"""
        pass


class EmpiricalReceptiveField():
    """
    A class to estimate empiral receptive field of a DNN model
    """
    def __init__(self, image_pixel_activation_estimator):
        """ image_pixel_activation_estimator is a ImagePixelActivation object """
        self.estimator = image_pixel_activation_estimator

    def compute(self, stim, layer, channel):
        """Generate RF based on provided image and pixel activation estimator """
        for s in stim:
            self.estimator(s,layer, channel)
            
            
class TheoreticalReceptiveField():
    def __init__(self, dnn):
        self.dnn = dnn
        
    def compute(self, layer, channel):
        pass


        