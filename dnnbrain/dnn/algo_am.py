import torch
import numpy as np

from torch.optim import Adam
from abc import ABC, abstractmethod


class Algorithm(ABC):
    """ 
    An Abstract Base Classes class to define interface for dnn algorithm 
    """
    def __init__(self, dnn, layer=None, channel=None):
        self.dnn = dnn
        self.layer = layer
        self.channel = channel
        
    def set_layer(self, layer, channel):
        self.layer = layer
        self.channel = channel


class SynthesisImage(Algorithm):
    """ 
    An Abstract Base Classes class to generate a synthetic image 
    that maximally activates a neuron
    """
    def __init__(self, dnn, layer=None, channel=None):
        """
        Parameter:
        ---------
        dnn[DNN]: dnnbrain's DNN object
        """
        super(SynthesisImage, self).__init__(dnn, layer, channel)
        self.dnn.eval()
        self.activation = None
        self.n_iter = None

    def set_layer(self, layer, channel):
        self.layer = layer
        self.channel = channel
        
    def set_params(self, n_iter=31):
        """
        Set the target

        Parameters:
        ----------
        layer[str]: layer name
        channel[int]: channel number
        """
        self.n_iter = n_iter
        
    def register_hooks(self):
        """
        Define register hook and register them to specific layer and channel.
        """
        def forward_hook(module, feat_in, feat_out):
            self.activation = feat_out[:, self.channel]
    
        # register forward hook to the target layer
        module = self.dnn.layer2module(self.layer)
        module.register_forward_hook(forward_hook)       
    
    @abstractmethod
    def synthesize(self):
        """
        Synthesize the image which maximally activates target layer and channel.         
        As this a abstract method, it is needed to be override in every subclass
        """


class L1SynthesisImage(SynthesisImage):
    """Use L1 regularization to estimate internal representation."""
    
    def synthesize(self):
        """
        Synthesize the image which maximally activates target layer and channel
        using L1 regularization.
        """
        # Hook the selected layer
        self.register_hooks()
        # Generate a random image
        image_size = (3,) + self.dnn.img_size
        optimal_image = np.uint8(np.random.uniform(150, 180, image_size))
        optimal_image.requires_grad(True)
        
        # Define optimizer for the image
        optimizer = Adam([optimal_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, self.n_iter+1):
            optimizer.zero_grad()
            
            # Forward pass layer by layer until the target layer
            # to triger the hook funciton.
            self.dnn(optimal_image)
                
            # Loss function is the mean of the output of the selected filter
            # We try to maximize the mean of the output of that specific filter
            loss = -torch.mean(self.activation) + np.abs(optimal_image).sum()
            print('Iteration:', i, 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            
        # Return the optimized image
        return optimal_image
