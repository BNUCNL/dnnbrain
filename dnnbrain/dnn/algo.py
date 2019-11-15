#import some packages needed
import numpy as np
import math
import torch
from torch.nn import ReLU
from torch.optim import Adam
from abc import ABC, abstractmethod


class TheoreticalReceptiveField():
    def __init__(self, dnn):
        self.dnn = dnn
        
    def compute(self, layer, channel):
        pass
    
class EmpiricalReceptiveField():
    """
    A class to estimate empiral receptive field of a DNN model
    """
    def __init__(self, dnn):
        self.model = dnn
        self.estimator = None
     
    @abstractmethod
    def set_params(self, window, stride):
        """Set parameter for the estimator"""        
        
    def compute(self, stim, layer, channel):
        """Generate RF based on provided image and pixel activation estimator """
        for s in stim:
            self.estimator(s,layer, channel)

class UpSamplingERF(EmpiricalReceptiveField):
    """
    A class to estimate empiral receptive field of a DNN model
    """
    def __init__(self, dnn):
        self.model = dnn
        self.estimator = UpSamplingActivatioMap(dnn.model)
        
    def set_params(self, window, stride):
        self.estimator.set_params(window, stride)

class OccluderERF(EmpiricalReceptiveField):
    """
    A class to estimate empiral receptive field of a DNN model
    """
    def __init__(self, dnn):
        self.model = dnn
        self.estimator = OccluderDiscrepancyMap(dnn.model)
        
    def set_params(self, window, stride):
        self.estimator.set_params(window, stride)
        
        
class MinmalImage():
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """
    def __init__(self, dnn, optimization_criterion = None):
        """
        dnn is a DNN object
        """
        self.dnn = dnn
        self.criterion = optimization_criterion
    
    @abstractmethod
    def set_params(self):
        """Set parameter for the estimator"""        

        
    def compute(self, stim, layer, channel):
        """Generate minmal image for image listed in stim object """
        
class MinmalComponentImage(MinmalImage):
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """
    def __init__(self, dnn, optimization_criterion = None):
        """
        dnn is a DNN object
        """
        self.dnn = dnn
        self.decomposer = 'pca'
        self.criterion = optimization_criterion
        self.ncomp = 10
    
    @abstractmethod
    def set_params(self,ncomp):
        """Set parameter for the decomposer"""        
        self.ncomp = ncomp;


class MinmalParcelImage(MinmalImage):
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """
    def __init__(self, dnn, optimization_criterion = None):
        """
        dnn is a DNN object
        """
        self.dnn = dnn
        self.decomposer = 'SLIC'
        self.criterion = optimization_criterion
        self.nparcel = 10
    
    @abstractmethod
    def set_params(self, nparcel):
        """Set parameter for the decomposer"""
        self.nparcel = 10    
        
        
class SaliencyImage(ABC):
    """ 
    An Abstract Base Classes class to define interface for image decomposer, 
    which decompose an image into different parcels or components
    """
    def __init__(self, dnn):
        self.dnn = dnn
        self.bp = None

    @abstractmethod
    def set_params(self, is_smooth):
        """set params for bp"""
        self.bp.set_parms(is_smooth)
        
    def compute(self, stim, layer, channel):
        """you need to write code to compute the saliency map using bp"""
        for s in stim:
            self.bp.gradient(s)
        

class VanlinaSaliencyImage(ABC):
    """ 
    An Abstract Base Classes class to define interface for image decomposer, 
    which decompose an image into different parcels or components
    """
    def __init__(self, dnn):
        self.dnn = dnn
        self.bp = VanilaBackPropGradient(dnn.model)

    @abstractmethod
    def set_params(self, is_smooth):
        """set params for bp"""
        self.bp.set_parms(is_smooth)
        

class GuidedSaliencyImage(ABC):
    """ 
    An Abstract Base Classes class to define interface for image decomposer, 
    which decompose an image into different parcels or components
    """
    def __init__(self, dnn):
        self.dnn = dnn
        self.bp = GuidedBackPropGradient(dnn.model)

    @abstractmethod
    def set_params(self, is_smooth):
        """set params for bp"""
        self.bp.set_parms(is_smooth)


class SynthesisImage(ABC):
    """ Use L1 regularization to estimate internel representation """
    def __init__(self, dnn):
        self.dnn = dnn
        self.am = None
        
    @abstractmethod
    def set_params(self):
        """set params for the am algorithm """
    
    def compute(self, stim, layer, channel):
        for s in stim:
            self.am.synthesize(s,layer,channel)
    

class L1SynthesisImage(ABC):
    """ Use L1 regularization to estimate internel representation """
    def __init__(self, dnn):
        self.dnn = dnn
        self.am = L1ActivationMaximization(dnn.model)
        
    def set_params(self, alpha):
        """set params for the am algorithm """
        self.am.set_params(alpha)


    

