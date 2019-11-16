#import some packages needed
from abc import ABC, abstractmethod


class ImagePixelActivation(ABC):
    """ 
    An Abstract Base Classes class to define interface for pixel activation, 
    which compute the activation for each pixel from an image
    """
    def __init__(self, dnn, metric = None):
        """metric [str]: the metric used to dervie the activation of the pixel """
        self.dnn = dnn
        self.metric = metric
        
    @abstractmethod
    def set_params(self):
        """ set parames """
        
    @abstractmethod
    def compute(self, image, layer, channel): 
        """The method use _estimate to caculate the pixel activtion"""

class OccluderDiscrepancyMap(ImagePixelActivation):
    def __init__(self, dnn, metric = 'max', window = (11, 11), stride = 2):
        """metric: max, min, L1, L2"""
        super(OccluderDiscrepancyMap, self).__init__(dnn, metric)
        self.metric = metric
        self.window = window
        self.stride = stride

        
    def set_params(self, window, stride, metric):
        """Set necessary parameters for the estimator"""
        self.window = window
        self.stride = stride
        self.metric = metric
    
    def compute(image, layer, channel):
        """ The method do real computation for discrepancy map based on sliding occluder"""
        pass

class UpsamplingActivationMap(ImagePixelActivation):
    def __init__(self, dnn, metric = 'bilinear'):
        """metric: bilinear"""
        super(UpsamplingActivationMap, self).__init__(dnn, metric)

    
    def set_params(self, metric):
        """Set necessary parameters for the estimator"""
        self.metric = metric
    
    def compute(image, layer, channel):
        """ The method do real computation for pixel activation based on feature mapping upsampling"""
        pass


class EmpiricalReceptiveField():
    """
    A class to estimate empiral receptive field of a DNN model
    """
    def __init__(self, pixel_activation_estimator):
        """ image_pixel_activation_estimator is a ImagePixelActivation object """
        self.activation_estimator = pixel_activation_estimator

    def compute(self, stim, layer, channel):
        """Generate RF based on provided image and pixel activation estimator """
        for s in stim:
            self.activation_estimator(s,layer, channel)
            
class TheoreticalReceptiveField():
    def __init__(self, dnn):
        self.dnn = dnn
        
    def compute(self, layer, channel):
        pass


        