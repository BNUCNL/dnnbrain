#import some packages needed
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

class OccluderDiscrepancyMap(Algorithm):      
    """ 
    An class to compute activation for each pixel from an image 
    using slide Occluder
    """
    def set_params(self, window=(11, 11), stride=2, metric='max'):
        """Set necessary parameters for the estimator"""
        self.window = window
        self.stride = stride
        self.metric = metric
    
    def compute(image):
        """ 
        Please implement implement the sliding occluder algothrim 
        for discrepancy map"""
        pass

class UpsamplingActivationMap(Algorithm):
    """ 
    A class to compute activation for each pixel from an image by upsampling 
    activation map
    """
    def set_params(self, metric='bilinear'):
        """Set necessary parameters for upsampling estimator"""
        self.metric = metric
    
    def compute(image):
        """ The method do real computation for pixel activation based on feature mapping upsampling"""
        pass

class EmpiricalReceptiveField():
    """
    A class to estimate empirical receptive field of a DNN model
    """
    def __init__(self, dnn, layer=None, channel=None):
        self.dnn = dnn
        self.layer = layer
        self.channel = channel
    
    @abstractmethod
    def set_params(self):
        """ set parames """
        
    @abstractmethod
    def compute(self, images):
        """Generate RF based on provided image and pixel activation estimator """
        pass
            

class OccluderERF(EmpiricalReceptiveField):
    """
    A class to estimate empiral receptive field of a DNN model
    """
    def __init__(self, dnn, layer, channel):
        """ image_pixel_activation_estimator is a Algorithm object """
        super(OccluderERF,self).__init__(dnn, layer, channel)      
        self. activation_estimator = \
        OccluderDiscrepancyMap(self.dnn, self.layer, self.channel)
        
    def set_params(self, window, stride, metric):
        self. activation_estimator.set_params(window, stride, metric)
    
    def compute(self, images):
        """Generate RF based on provided image and pixel activation estimator """
        pass
    
class UpsamplingERF(EmpiricalReceptiveField):
    """
    A class to estimate empiral receptive field of a DNN model
    """
    def __init__(self, dnn, layer, channel):
        super(UpsamplingERF,self).__init__(dnn, layer, channel)      
        self. activation_estimator = \
        UpsamplingActivationMap(self.dnn, self.layer, self.channel)
 
    def set_params(self, metric='bilinear'):
        self. activation_estimator.set_params(metric)
    
    def compute(self, images):
        """Generate RF based on provided image and pixel activation estimator """
        pass
    
class TheoreticalReceptiveField():
    def __init__(self, dnn):
        self.dnn = dnn
        
    def compute(self, layer, channel):
        pass


        