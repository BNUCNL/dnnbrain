import abc
import torch
from torch.nn import ReLU


class SaliencyImage(abc.ABC):
    """ 
    An Abstract Base Classes class to define interfaces for gradient backpropagation
    """
    def __init__(self, dnn):

        self.dnn = dnn
        self.dnn.model.eval()
        self.layer = None
        self.channel = None
        self.activation = []
        self.gradient = None
        
    def set(self, layer, channel=None):
        self.layer = layer
        self.channel = channel
    
    @abc.abstractmethod
    def register_hooks(self):
        """
        Define register hook and register them to specific layer and channel.
        As this a abstract method, it is needed to be override in every subclass
        """

    def gradient(self, image):
        """
        Compute gradient with back propagation algorithm
        """
        self.register_hooks()
        # Forward
        self.dnn.model(image)
        # Zero grads
        self.dnn.model.zero_grad()
        # Backward pass
        self.activation.pop().backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradient = self.gradients.data.numpy()[0]
        return gradient
    
    def smooth_gradient(self, image): 
        """
        Compute smoothed gradient. 
        It will use the gradient method to compute the gradient and then smooth it
        """
        pass
        

class VanlinaSaliencyImage(BackPropGradient):
    """ 
    A class to compute vanila Backprob gradient for a image.
    """

    def register_hooks(self):
        """
        Override the abstract method from BackPropGradient class to
        define a specific hook for vanila backprop gradient. 
        """  
        def forward_hook(module, feat_in, feat_out):
            self.activation.append(feat_out[:, self.channel])

        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_in[0]
        
        # register forward hook to the target layer
        
        # register backward to the first layer


class GuidedSaliencyImage(BackPropGradient):
    """ 
    A class to compute Guided Backprob gradient for a image.

    """
    
    def register_hooks(self):
        """
        Override the abstract method from BackPropGradient class to
        define a specific hook for guided backprop gradient. 
        """  
        def first_layer_backward_hook(module, grad_in, grad_out):
            self.gradient = grad_in[0]

        def targ_layer_forward_hook(module, feat_in, feat_out):
            self.activation.append(feat_out[:, self.channel])

        def forward_hook(module, feat_in, feat_out):
            self.activation.append(feat_out)

        def backward_hook(module, grad_in, grad_out):
            act = self.activation.pop()
            act[act > 0] = 1
            grad = torch.clamp(grad_out[0], min=0.)
            grad = grad * act
            return (grad,)
        
        # register hook for target module
        targ_module = self.dnn.model
        for L in self.layer:
            targ_module = targ_module._modules[L]
        targ_module.register_forward_hook(targ_layer_forward_hook)
        targ_module.register_backward_hook(backward_hook)
        
        # register forward and backward hook to all relu layers before targe layer
        modules = list(self.dnn.model.features.named_children())
        for name,module in modules:
            if module is not targ_module:
                if isinstance(module, ReLU):
                    module.register_forward_hook(forward_hook)
                    module.register_backward_hook(backward_hook)
            else:         
                break

        # register backward to the first layer
        first_layer = self.dnn.model[0][1]
        first_layer.register_backward_hook(first_layer_backward_hook)

