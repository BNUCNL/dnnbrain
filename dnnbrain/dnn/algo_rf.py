#import some packages needed
from abc import ABC, abstractmethod


class ImagePixelActivation(ABC):
    """ 
    An Abstract Base Classes class to define interface for pixel activation, 
    which compute the activation for each pixel from an image
    """
    def __init__(self, dnn, layer = None, channel = None):
        """metric [str]: the metric used to dervie the activation of the pixel """
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
        """The method use _estimate to caculate the pixel activtion"""

class OccluderDiscrepancyMap(ImagePixelActivation):      
    
    def set_params(self, window=(11, 11), stride=2, metric='max'):
        """Set necessary parameters for the estimator"""
        self.window = window
        self.stride = stride
        self.metric = metric
    
    def compute(image):
        """ The method do real computation for discrepancy map based on sliding occluder"""
        pass

class UpsamplingActivationMap(ImagePixelActivation):
    
    def set_params(self, metric='bilinear'):
        """Set necessary parameters for upsampling estimator"""
        self.metric = metric
    
    def compute(image):
        """ The method do real computation for pixel activation based on feature mapping upsampling"""
        pass

class EmpiricalReceptiveField():
    """
    A class to estimate empiral receptive field of a DNN model
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
        """ image_pixel_activation_estimator is a ImagePixelActivation object """
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
        """ image_pixel_activation_estimator is a ImagePixelActivation object """
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


        