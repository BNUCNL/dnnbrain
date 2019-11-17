import abc
import torch

from torch import nn


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


if __name__ == '__main__':



