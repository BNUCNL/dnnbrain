import torch
from torch.nn import ReLU
from abc import ABC, abstractmethod

class BackPropGradient(ABC):
    """ 
    An Abstract Base Classes class to define interface for image decomposer, 
    which decompose an image into different parcels or components
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.activation = []
        self.gradient = None
        
    def set_layer(self, layer = None, channel = None):
        self.layer = layer
        self.channel = channel
    
    @abstractmethod
    def register_hooks(self):
        """
        Define regsister hook and register them to specific layer and channel.
        As this a abstract method, it is needed to be override in every childclass
        """

    def gradient(self, image):
        """
        Compute gradient with back propgation algorithm 
        """
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
        """
        Compute smoothed gradient. 
        It will use the graident method to compute the gradient and then smooth it
        """
        pass
        

class VanilaBackPropGradient(BackPropGradient):
    """ 
    A class to compute vanila Backprob gradient for a image.
    """
    def register_hooks(self):
        """
        Overrie the abstract method from BackPropGradient class to 
        define a specific hook for vanila backprop gradient. 
        """  
        def forward_hook(module, feat_in,feat_out):
            self.activation.append(feat_out[:, self.channel])
        def backward_hook(module,grad_in,grad_out):
            self.gradient = grad_in[0]
        
        # register forward hook to the target layer
        module = self.model
        for L in self.layer:
            module = module._modules[L]
        module.register_forward_hook(forward_hook)
        
        # register backward to the first layer
        first_layer = self.model[0][1]
        first_layer.register_backward_hook(backward_hook)

class GuidedBackPropGradient(BackPropGradient):
    """ 
    A class to compute Guided Backprob gradient for a image.

    """    
    
    def register_hooks(self):
        """
        Overrie the abstract method from BackPropGradient class to 
        define a specific hook for guided backprop gradient. 
        """  
        def first_layer_backward_hook(module,grad_in,grad_out):
            self.out_img = grad_in[0]
        def targ_layer_forward_hook(module,feat_in,feat_out):
            self.activation.append(feat_out[:, self.channel])
        def forward_hook(module,feat_in,feat_out):
            self.activation.append(feat_out)
            # = feat_out[:, self.channel]
        def backward_hook(module,grad_in,grad_out):
            act = self.activation.pop()
            act[act > 0] = 1
            grad = torch.clamp(grad_out[0],min = 0.)
            grad = grad * act
            return (grad,)
        
        # register hook for target module 
        targ_module = self.model
        for L in self.layer:
            targ_module = targ_module._modules[L]
        targ_module.register_forward_hook(targ_layer_forward_hook)
        targ_module.register_backward_hook(backward_hook) 
        
        # register forward and backward hook to all layers bellow targe layer
        modules = list(self.model.features.named_children())
        for name,module in modules:
            if module is not targ_module:
                if isinstance(module, ReLU):
                    module.register_forward_hook(forward_hook)
                    module.register_backward_hook(backward_hook)
            else:         
                break


        # register backward to the first layer
        first_layer = self.model[0][1]
        first_layer.register_backward_hook(first_layer_backward_hook)

             

