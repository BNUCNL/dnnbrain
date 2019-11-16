import abc
import torch
from torch.nn import ReLU


class SaliencyImage(abc.ABC):
    """ 
    An Abstract Base Classes class to define interfaces for gradient back propagation
    """
    def __init__(self, dnn, first_layer):
        """
        Parameter:
        ---------
        dnn[DNN]: dnnbrain's DNN object
        """
        self.dnn = dnn
        self.dnn.eval()
        self.first_layer = first_layer
        self.target_layer = None
        self.channel = None
        self.activation = []
        self.gradient = None
        
    def set(self, layer, channel):
        """
        Set the target

        Parameters:
        ----------
        layer[str]: layer name
        channel[int]: channel number
        """
        self.target_layer = layer
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

        Parameter:
        ---------
        image[Tensor]: an input of the model, with shape as (1, n_chn, n_height, n_width)

        Return:
        ------
        gradient[ndarray]: the input's gradients corresponding to the target activation
            with shape as (n_chn, n_height, n_width)
        """
        self.register_hooks()
        # Forward
        self.dnn(image)
        # Zero grads
        self.dnn.model.zero_grad()
        # Backward pass
        self.activation.pop().backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradient = self.gradient.data.numpy()[0]
        return gradient
    
    def smooth_gradient(self, image): 
        """
        Compute smoothed gradient. 
        It will use the gradient method to compute the gradient and then smooth it
        """
        pass
        

class VanlinaSaliencyImage(SaliencyImage):
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
        trg_module = self.dnn.layer2module(self.target_layer)
        trg_module.register_forward_hook(forward_hook)
        
        # register backward to the first layer
        first_module = self.dnn.layer2module(self.first_layer)
        first_module.register_backward_hook(backward_hook)


class GuidedSaliencyImage(SaliencyImage):
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
        trg_module = self.dnn.layer2module(self.target_layer)
        trg_module.register_forward_hook(targ_layer_forward_hook)
        # trg_module.register_backward_hook(backward_hook)
        
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
        first_layer = self.dnn.layer2module(self.first_layer)
        first_layer.register_backward_hook(first_layer_backward_hook)
