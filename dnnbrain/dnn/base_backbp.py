import numpy as np
import math
import torch
from torch.nn import ReLU
from torch.optim import Adam
from abc import ABC, abstractmethod
try:
    from misc_functions import preprocess_image, recreate_image
except ModuleNotFoundError:
    pass
    #raise Exception('Please install misc_functions in your work station')
    

class BackPropGradient(ABC):
    """ 
    An Abstract Base Classes class to define interface for image decomposer, 
    which decompose an image into different parcels or components
    """
    def __init__(self, model):
        self.model = model
        self.activation = None
        self.gradient = None
        
    def set_layer(self, layer = None, channel = None):
        self.layer = layer
        self.channel = channel
    
    @abstractmethod
    def register_hooks(self):
        return NotImplemented

    def gradient(self, image):
        self.register_hooks()
        # Forward
        self.model(image)
        # Zero grads
        self.model.zero_grad()
        # Backward pass
        self.activation.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradient = self.gradients.data.numpy()[0]
        return gradient
    
    def smooth_gradient(self, image): 
        'Please do your implementation here'
        pass

class VanilaBackPropGradient(BackPropGradient):
    """ 
    A class to compute vanila Backprob gradient for a image.
    """
    
    def register_hooks(self):
        def forward_hook(module, input_feature,output_feature):
            self.activation = output_feature[:, self.channel]
        def backward_hook(module,grad_in,grad_out):
            self.gradient = grad_out[0]
            
        module = self.model
        for k in self.layer:
            module = module._modules[k]
            
        module.register_forward_hook(forward_hook)

        first_layer = self.model[0][1]
        first_layer.register_backward_hook(backward_hook)

          

        
    
class GuidedBackPropGradient(BackPropGradient):
    """ 
    A class to compute Guided Backprob gradient for a image.

    """    
    
    def register_hooks(self):
        def register_first_layer_hook(module,grad_in,grad_out):
            self.out_img = grad_in[0]
        def forward_hook(module,input_feature,output_feature):
            self.activation = output_feature[:, self.channel]
        def backward_hook(module,grad_in,grad_out):
            grad = self.activation.pop()
            grad[grad > 0] = 1
            g_positive = torch.clamp(grad_out[0],min = 0.)
            result_grad = grad * g_positive
            return (result_grad,)
        
        targ_module = self.model
        for k in self.layer:
            targ_module = targ_module._modules[k]
            
            
        modules = list(self.model.features.named_children())
        for name,module in modules:
            if module is targ_module:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

            if isinstance(module,nn.ReLU):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
        
    
        first_layer = self.model[0][1]
        first_layer.register_backward_hook(register_first_layer_hook)

             

