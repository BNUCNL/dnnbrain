import abc
import torch

import numpy as np


class Algorithm(abc.ABC):
    """
    An Abstract Base Classes class to define interface for dnn algorithm
    """
    def __init__(self, dnn, layer=None, channel=None):
        """
        Parameters:
        ----------
        dnn[DNN]: dnnbrain's DNN object
        layer[str]: name of the layer where the algorithm performs on
        channel[int]: sequence number of the channel where the algorithm performs on
        """
        self.dnn = dnn
        self.dnn.eval()
        self.layer = layer
        self.channel = channel

    def set_layer(self, layer, channel):
        """
        Set layer or its channel

        Parameters:
        ----------
        layer[str]: name of the layer where the algorithm performs on
        channel[int]: sequence number of the channel where the algorithm performs on
        """
        self.layer = layer
        self.channel = channel


class SaliencyImage(Algorithm):
    """
    An Abstract Base Classes class to define interfaces for gradient back propagation
    """

    def __init__(self, dnn, from_layer=None, from_chn=None):
        """
        Parameters:
        ----------
        dnn[DNN]: dnnbrain's DNN object
        from_layer[str]: name of the layer where gradients back propagate from
        from_chn[int]: sequence number of the channel where gradient back propagate from
        """
        super(SaliencyImage, self).__init__(dnn, from_layer, from_chn)

        self.to_layer = None
        self.activation = []
        self.gradient = None
        self.hook_handles = []

    @abc.abstractmethod
    def register_hooks(self):
        """
        Define register hook and register them to specific layer and channel.
        As this a abstract method, it is needed to be override in every subclass
        """

    def backprop(self, image, to_layer=None):
        """
        Compute gradients of the layer corresponding to the self.layer and self.channel
        by back propagation algorithm.

        Parameters:
        ----------
        image[Tensor]: an input of the model, with shape as (1, n_chn, n_height, n_width)
        to_layer[str]: name of the layer where gradients back propagate to
            If is None, get the first layer in the layers recorded in DNN.

        Return:
        ------
        gradient[ndarray]: gradients of the to_layer with shape as (n_chn, n_row, n_col)
            If layer is the first layer of the model, its shape is (n_chn, n_height, n_width)
        """
        # deal with parameters
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("The input data must be a tensor with shape as "
                             "(1, n_chn, n_height, n_width)")
        self.to_layer = self.dnn.layers[0] if to_layer is None else to_layer

        self.register_hooks()
        # forward
        image.requires_grad_(True)
        self.dnn(image)
        # zero grads
        self.dnn.model.zero_grad()
        # backward
        self.activation.pop().backward()
        # tensor to ndarray
        # [0] to get rid of the first dimension (1, n_chn, n_row, n_col)
        gradient = self.gradient.data.numpy()[0]
        self.gradient = None

        # remove hooks
        for hook_handle in self.hook_handles:
            hook_handle.remove()

        return gradient

    def backprop_smooth(self, image, n_iter, sigma_multiplier=1, to_layer=None):
        """
        Compute smoothed gradient.
        It will use the gradient method to compute the gradient and then smooth it

        Parameters:
        ----------
        image[Tensor]: an input of the model, with shape as (1, n_chn, n_height, n_width)
        n_iter[int]: the number of noisy images to be generated before average.
        sigma_multiplier[int]: multiply when calculating std of noise
        to_layer[str]: name of the layer where gradients back propagate to
            If is None, get the first layer in the layers recorded in DNN.

        Return:
        ------
        gradient[ndarray]: gradients of the to_layer with shape as (n_chn, n_row, n_col)
            If layer is the first layer of the model, its shape is (n_chn, n_height, n_width)
        """
        # deal with parameters
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("The input data must be a tensor with shape as "
                             "(1, n_chn, n_height, n_width)")
        assert isinstance(n_iter, int) and n_iter > 0, \
            'The number of iterations must be a positive integer!'
        self.to_layer = self.dnn.layers[0] if to_layer is None else to_layer

        self.register_hooks()
        gradient = 0
        sigma = sigma_multiplier / (image.max() - image.min()).item()
        for iter_idx in range(1, n_iter+1):
            # prepare image
            image_noisy = image + image.normal_(0, sigma**2)
            image_noisy.requires_grad_(True)

            # forward
            self.dnn(image_noisy)
            # clean old gradients
            self.dnn.model.zero_grad()
            # backward
            self.activation.pop().backward()
            # tensor to ndarray
            # [0] to get rid of the first dimension (1, n_chn, n_row, n_col)
            gradient += self.gradient.data.numpy()[0]
            print(f'Finish: noisy_image{iter_idx}/{n_iter}')

        # remove hooks
        for hook_handle in self.hook_handles:
            hook_handle.remove()

        self.gradient = None
        gradient = gradient / n_iter
        return gradient


class VanillaSaliencyImage(SaliencyImage):
    """
    A class to compute vanila Backprob gradient for a image.
    """

    def register_hooks(self):
        """
        Override the abstract method from BackPropGradient class to
        define a specific hook for vanila backprop gradient.
        """

        def forward_hook(module, feat_in, feat_out):
            self.activation.append(torch.mean(feat_out[0, self.channel-1]))

        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_in[0]

        # register forward hook to the target layer
        module = self.dnn.layer2module(self.layer)
        self.hook_handles.append(module.register_forward_hook(forward_hook))

        # register backward to the first layer
        to_module = self.dnn.layer2module(self.to_layer)
        self.hook_handles.append(to_module.register_backward_hook(backward_hook))
