#import some packages needed
import numpy as np
import torch
from torch.optim import Adam
from abc import ABC, abstractmethod


class ActivationMaximization(ABC):
    """ 
    An Abstract Base Classes class to generate a synthetic image 
    that maximally activates a neuron
    """
    def __init__(self, model = None):
        self.model = model
        self.model.eval()
        self.activation = None
        self.channel = None
        self.layer = []
        self.niter = None
        
    def set_layer(self, layer, channel, niter = None):
        self.layer = layer
        self.channel = channel
        self.niter = 31
    
    def set_image_size(self, xsize, ysize):
        self.image_size = (xsize, ysize, 3)
        
    def register_hooks(self):
        """
        Define regsister hook and register them to specific layer and channel.
        As this a abstract method, it is needed to be override in every childclass
        """
        def forward_hook(module, feat_in,feat_out):
            self.activation.append(feat_out[:, self.channel])
    
        # register forward hook to the target layer
        module = self.model
        for L in self.layer:
            module = module._modules[L]
        module.register_forward_hook(forward_hook)       
    
    @abstractmethod
    def synthesize(self, layer, channel):
        """
        Synthesize the image which maximally activates target layer and channel.         
        As this a abstract method, it is needed to be override in every childclass
        """

class L1ActivationMaximization(ActivationMaximization):
    """ Use L1 regularization to estimate internel representation """
    
    def synthesize(self, layer, channel):
        """
        Synthesize the image which maximally activates target layer and channel
        using L1 regularization.
        """
        # Hook the selected layer
        self.register_hooks()
        # Generate a random image
        optimal_image = np.uint8(np.random.uniform(150, 180, self.image_size))

        # Define optimizer for the image
        optimizer = Adam([optimal_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, self.niter):
            optimizer.zero_grad()
            
            # Forward pass layer by layer until the target layer
            # to triger the hook funciton.
            forawrd_image = optimal_image
            for name, module in enumerate(self.model):
                forawrd_image = module(forawrd_image)
                if name == self.layer:
                    break
                
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.activation) + np.abs(optimal_image).sum()
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            
        # Return the optimized image
        return optimal_image


   