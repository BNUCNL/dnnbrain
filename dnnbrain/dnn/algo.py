import abc
import torch
import numpy as np

from torch.optim import Adam
from dnnbrain.dnn.core import Mask
from dnnbrain.dnn.base import ip


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
        if np.logical_xor(layer is None, channel is None):
            raise ValueError("layer and channel must be used together!")
        if layer is not None:
            self.set_layer(layer, channel)
        self.dnn = dnn
        self.dnn.eval()

    def set_layer(self, layer, channel):
        """
        Set layer or its channel

        Parameters:
        ----------
        layer[str]: name of the layer where the algorithm performs on
        channel[int]: sequence number of the channel where the algorithm performs on
        """
        self.mask = Mask()
        self.mask.set(layer, channels=[channel])

    def get_layer(self):
        """
        Get layer or its channel

        Parameters:
        ----------
        layer[str]: name of the layer where the algorithm performs on
        channel[int]: sequence number of the channel where the algorithm performs on
        """
        layer = self.mask.layers[0]
        channel = self.mask.get(layer)['chn'][0]
        return layer, channel


class SaliencyImage(Algorithm):
    """
    An Abstract Base Classes class to define interfaces for gradient back propagation
    Note: the saliency image values are not applied with absolute operation.
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
        self.activation = None
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
        Compute gradients of the to_layer corresponding to the from_layer and from_channel
        by back propagation algorithm.

        Parameters:
        ----------
        image[ndarray|Tensor|PIL.Image]: image data
        to_layer[str]: name of the layer where gradients back propagate to
            If is None, get the first layer in the layers recorded in DNN.

        Return:
        ------
        gradient[ndarray]: gradients of the to_layer with shape as (n_chn, n_row, n_col)
            If layer is the first layer of the model, its shape is (3, n_height, n_width)
        """
        # register hooks
        self.to_layer = self.dnn.layers[0] if to_layer is None else to_layer
        self.register_hooks()

        # forward
        image = self.dnn.test_transform(ip.to_pil(image))
        image = image.unsqueeze(0)
        image.requires_grad_(True)
        self.dnn(image)
        # zero grads
        self.dnn.model.zero_grad()
        # backward
        self.activation.backward()
        # tensor to ndarray
        # [0] to get rid of the first dimension (1, n_chn, n_row, n_col)
        gradient = self.gradient.data.numpy()[0]

        # remove hooks
        for hook_handle in self.hook_handles:
            hook_handle.remove()

        # renew some attributions
        self.activation = None
        self.gradient = None

        return gradient

    def backprop_smooth(self, image, n_iter, sigma_multiplier=0.1, to_layer=None):
        """
        Compute smoothed gradient.
        It will use the gradient method to compute the gradient and then smooth it

        Parameters:
        ----------
        image[ndarray|Tensor|PIL.Image]: image data
        n_iter[int]: the number of noisy images to be generated before average.
        sigma_multiplier[int]: multiply when calculating std of noise
        to_layer[str]: name of the layer where gradients back propagate to
            If is None, get the first layer in the layers recorded in DNN.

        Return:
        ------
        gradient[ndarray]: gradients of the to_layer with shape as (n_chn, n_row, n_col)
            If layer is the first layer of the model, its shape is (n_chn, n_height, n_width)
        """
        assert isinstance(n_iter, int) and n_iter > 0, \
            'The number of iterations must be a positive integer!'

        # register hooks
        self.to_layer = self.dnn.layers[0] if to_layer is None else to_layer
        self.register_hooks()

        image = self.dnn.test_transform(ip.to_pil(image))
        image = image.unsqueeze(0)
        gradient = 0
        sigma = sigma_multiplier * (image.max() - image.min()).item()
        for iter_idx in range(1, n_iter+1):
            # prepare image
            image_noisy = image + image.normal_(0, sigma**2)
            image_noisy.requires_grad_(True)

            # forward
            self.dnn(image_noisy)
            # clean old gradients
            self.dnn.model.zero_grad()
            # backward
            self.activation.backward()
            # tensor to ndarray
            # [0] to get rid of the first dimension (1, n_chn, n_row, n_col)
            gradient += self.gradient.data.numpy()[0]
            print(f'Finish: noisy_image{iter_idx}/{n_iter}')

        # remove hooks
        for hook_handle in self.hook_handles:
            hook_handle.remove()

        # renew some attributions
        self.activation = None
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
        from_layer, from_chn = self.get_layer()

        def from_layer_acti_hook(module, feat_in, feat_out):
            self.activation = torch.mean(feat_out[0, from_chn-1])

        def to_layer_grad_hook(module, grad_in, grad_out):
            self.gradient = grad_in[0]

        # register forward hook to the target layer
        from_module = self.dnn.layer2module(from_layer)
        from_handle = from_module.register_forward_hook(from_layer_acti_hook)
        self.hook_handles.append(from_handle)

        # register backward to the first layer
        to_module = self.dnn.layer2module(self.to_layer)
        to_handle = to_module.register_backward_hook(to_layer_grad_hook)
        self.hook_handles.append(to_handle)


class GuidedSaliencyImage(SaliencyImage):
    """
    A class to compute Guided Backprob gradient for a image.
    """

    def register_hooks(self):
        """
        Override the abstract method from BackPropGradient class to
        define a specific hook for guided backprop gradient.
        """
        from_layer, from_chn = self.get_layer()

        def from_layer_acti_hook(module, feat_in, feat_out):
            self.activation = torch.mean(feat_out[0, from_chn - 1])

        def to_layer_grad_hook(module, grad_in, grad_out):
            self.gradient = grad_in[0]

        def relu_grad_hook(module, grad_in, grad_out):
            grad_in[0][grad_out[0] <= 0] = 0

        # register hook for from_layer
        from_module = self.dnn.layer2module(from_layer)
        handle = from_module.register_forward_hook(from_layer_acti_hook)
        self.hook_handles.append(handle)

        # register backward hook to all relu layers util from_layer
        for module in self.dnn.model.modules():
            # register hooks for relu
            if isinstance(module, torch.nn.ReLU):
                handle = module.register_backward_hook(relu_grad_hook)
                self.hook_handles.append(handle)

            if module is from_module:
                break

        # register hook for to_layer
        to_module = self.dnn.layer2module(self.to_layer)
        handle = to_module.register_backward_hook(to_layer_grad_hook)
        self.hook_handles.append(handle)


class SynthesisImage(Algorithm):
    """
    Generate a synthetic image that maximally activates a neuron.
    """

    def __init__(self, dnn, layer, channel,
                 activ_metric='mean', regular_metric=None):
        """
        Parameters:
        ----------
        dnn[DNN]: DNNBrain DNN
        layer[str]: name of the layer where the algorithm performs on
        channel[int]: sequence number of the channel where the algorithm performs on
        activ_metric[str]: The metric method to summarize activation
        regular_metric[str]: The metric method of regularization
        """
        super(SynthesisImage, self).__init__(dnn, layer, channel)
        self.set_metric(activ_metric, regular_metric)
        self.activation = None
        self.optimal_image = None

        # loss recorder
        self.activation_loss = []
        self.regularization_loss = []

    def set_metric(self, activ_metric, regular_metric):
        """
        Set metric methods

        Parameter:
        ---------
        activ_metric[str]: The metric method to summarize activation
        regular_metric[str]: The metric method of regularization
        """
        # activation metric setting
        if activ_metric == 'max':
            self.activ_metric = self.max_activation
        elif activ_metric == 'mean':
            self.activ_metric = self.mean_activation
        else:
            raise AssertionError('Only max and mean activation metrics are supported')

        # regularization metric setting
        if regular_metric is None:
            self.regular_metric = None
        elif regular_metric == 'L1':
            self.regular_metric = self.L1_norm
        else:
            raise AssertionError('Only L1 is supported')

    def mean_activation(self):
        activ = -torch.mean(self.activation)
        self.activation_loss.append(activ)
        return activ

    def max_activation(self):
        activ = -torch.max(self.activation)
        self.activation_loss.append(activ)
        return activ

    def L1_norm(self):
        reg = np.abs(self.optimal_image.detach().numpy()).sum()
        self.regularization_loss.append(reg)

        return reg

    def total_variation(self):
        pass

    def gaussian_blur(self):
        pass

    def mean_image(self):
        pass

    def center_bias(self):
        pass

    def register_hooks(self):
        """
        Define register hook and register them to specific layer and channel.
        """
        layer, chn = self.get_layer()

        def forward_hook(module, feat_in, feat_out):
            self.activation = feat_out[0, chn]

        # register forward hook to the target layer
        module = self.dnn.layer2module(layer)
        module.register_forward_hook(forward_hook)

    def synthesize(self, n_iter=30):
        """
        Synthesize the image which maximally activates target layer and channel

        Parameter:
        ---------
        n_iter[int]: the number of iterations

        Return:
        ------
            [ndarray]: the synthesized image with shape as (n_chn, height, width)
        """
        # Hook the selected layer
        self.register_hooks()

        # Generate a random image
        image = np.random.randint(0, 256, (3, *self.dnn.img_size)).astype(np.uint8)
        image = ip.to_pil(image)
        self.optimal_image = self.dnn.test_transform(image).unsqueeze(0)
        self.optimal_image.requires_grad_(True)

        # Define optimizer for the image
        self.activation_loss = []
        self.regularization_loss = []
        optimizer = Adam([self.optimal_image], lr=0.1, betas=(0.9, 0.99))
        for i in range(1, n_iter + 1):
            # clear gradients for next train
            optimizer.zero_grad()

            # Forward pass layer by layer until the target layer
            # to triger the hook funciton.
            self.dnn.model(self.optimal_image)

            # computer loss
            alpha = 0.1
            if self.regular_metric is None:
                loss = self.activ_metric()
            else:
                loss = self.activ_metric() + alpha * self.regular_metric()

            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            print(f'Iteration: {i}/{n_iter}')

        # Return the optimized image
        return self.optimal_image[0].detach().numpy()
