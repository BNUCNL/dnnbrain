import abc
import copy
import torch
import numpy as np

from torch.optim import Adam
from os.path import join as pjoin
from dnnbrain.dnn.core import Mask
from dnnbrain.dnn.base import ip
from PIL import ImageFilter
from skimage import filters


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

    def __init__(self, dnn, layer=None, channel=None,
                 activ_metric='mean', regular_metric=None, precondition_metric=None,
                 save_out_interval=False,print_inter_loss=False,):
        """
        Parameters:
        ----------
        dnn[DNN]: DNNBrain DNN
        layer[str]: name of the layer where the algorithm performs on
        channel[int]: sequence number of the channel where the algorithm performs on
        activ_metric[str]: The metric method to summarize activation
        regular_metric[str]: The metric method of regularization
        precondition_metric[str]: The metric method of precondition
        """
        super(SynthesisImage, self).__init__(dnn, layer, channel)
        self.set_metric(activ_metric, regular_metric, precondition_metric, save_out_interval,print_inter_loss)
        self.activ_loss = None
        self.optimal_image = None


        # loss recorder
        self.activ_losses = []
        self.regular_losses = []


    def set_metric(self, activ_metric, regular_metric,
                   precondition_metric,save_out_interval,print_inter_loss):
        """
        Set metric methods

        Parameter:
        ---------
        activ_metric[str]: The metric method to summarize activation
        regular_metric[str]: The metric method of regularization
        precondition_metric[str]: The metric method of preconditioning
        save_out_internal[str]: the method of saving the pics in interations
        """
        # activation metric setting
        if activ_metric == 'max':
            self.activ_metric = torch.max
        elif activ_metric == 'mean':
            self.activ_metric = torch.mean
        else:
            raise AssertionError('Only max and mean activation metrics are supported')

        # regularization metric setting
        if regular_metric is None:
            self.regular_metric = self._regular_default
        elif regular_metric == 'L1':
            self.regular_metric = self._L1_norm
        elif regular_metric == 'L2':
            self.regular_metric = self._L2_norm
        elif regular_metric == 'TV':
            self.regular_metric = self._total_variation
        else:
            raise AssertionError('Only L1, L2, and total variation are supported!')

        # precondition metric setting
        if precondition_metric is None:
            self.precondition_metric = self._precondition_default
        elif precondition_metric == 'GB':
            self.precondition_metric = self._gaussian_blur
        else:
            raise AssertionError('Only Gaussian Blur is supported!')

        # saving interval pics in iteration setting
        if save_out_interval is True:
            self.save_out_interval = self._save_out
        elif save_out_interval is False:
            self.save_out_interval = self._close_save

        # print interation loss
        if print_inter_loss is True:
            self.print_inter_loss = self._print_loss
        elif print_inter_loss is False:
            self.print_inter_loss = self._print_close

    def _print_loss(self,i,step,n_iter,loss):
        if i % step == 0:
            print(f'Interation: {i}/{n_iter}; Loss: {loss}')

    def _print_close(self,i,step,n_iter,loss):
        pass


    def _save_out(self,currti,save_interval,save_path):
        if (currti+1) % save_interval == 0:
            img_out = self.optimal_image[0].detach().numpy().copy()
            img_out = ip.to_pil(img_out, True)
            img_out.save(pjoin(save_path, f'synthesized_image_iter{currti+1}.jpg'))

    def _close_save(self, currti,save_interval, save_path):
        pass


    def _regular_default(self):
        reg = 0
        return reg

    def _L1_norm(self):
        reg = torch.abs(self.optimal_image).sum()
        self.regular_losses.append(reg.item())
        return reg

    def _L2_norm(self):
        reg = torch.sqrt(torch.sum(self.optimal_image ** 2))
        self.regular_losses.append(reg.item())
        return reg

    def _total_variation(self):
        # calculate the difference of neighboring pixel-values
        diff1 = self.optimal_image[0, :, 1:, :] - self.optimal_image[0, :, :-1, :]
        diff2 = self.optimal_image[0, :, :, 1:] - self.optimal_image[0, :, :, :-1]

        # calculate the total variation
        reg = torch.sum(torch.abs(diff1)) + torch.sum(torch.abs(diff2))
        self.regular_losses.append(reg.item())
        return reg

    def _precondition_default(self, GB_radius):
        precond_image = self.optimal_image[0].detach().numpy()
        precond_image = ip.to_tensor(precond_image).float()
        precond_image = copy.deepcopy(precond_image)
        return precond_image

    def _gaussian_blur(self, radius):
        precond_image = filters.gaussian(self.optimal_image[0].detach().numpy(), radius)
        precond_image = ip.to_tensor(precond_image).float()
        precond_image = copy.deepcopy(precond_image)
        return precond_image

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
            self.activ_loss = -self.activ_metric(feat_out[0, chn-1])
            self.activ_losses.append(self.activ_loss.item())

        # register forward hook to the target layer
        module = self.dnn.layer2module(layer)
        handle = module.register_forward_hook(forward_hook)

        return handle

    def synthesize(self, init_image=None, lr=0.1, regular_lambda=1, n_iter=30,
                   save_path=None, save_interval=None, GB_radius=None, step=1):
        """
        Synthesize the image which maximally activates target layer and channel

        Parameter:
        ---------
        init_image[ndarray|Tensor|PIL.Image]: initialized image
        lr[float]: learning rate
        regular_lambda[float]: the lambda of the regularization
        n_iter[int]: the number of iterations
        save_path[str]: the directory to save images
            If is None, do nothing.
            else, save synthesized image at the last iteration.
        save_interval[int]: save interval
            If is None, do nothing.
            else, save_path must not be None.
                Save out synthesized images per 'save interval' iterations.
        GB_radius[float]
        Return:
        ------
            [ndarray]: the synthesized image with shape as (n_chn, height, width)
        """
        # Hook the selected layer
        handle = self.register_hooks()

        # prepare initialized image
        if init_image is None:
            # Generate a random image
            init_image = np.random.rand(3,*self.dnn.img_size)
            init_image = ip.to_tensor(init_image).float()
            init_image = copy.deepcopy(init_image)
        else:
            init_image = ip.to_tensor(init_image).float()
            init_image = copy.deepcopy(init_image)

        self.activ_losses = []
        self.regular_losses = []

        # prepare for loss
        for i in range(n_iter):

            self.optimal_image = init_image.unsqueeze(0)
            self.optimal_image.requires_grad_(True)
            optimizer = Adam([self.optimal_image], lr=lr)

            # save out
            self.save_out_interval(i,save_interval,save_path)


            # Forward pass layer by layer until the target layer
            # to triger the hook funciton.
            self.dnn.model(self.optimal_image)

            # computer loss
            loss = self.activ_loss + regular_lambda * self.regular_metric()

            # zero gradients
            optimizer.zero_grad()
            # Backward

            loss.backward()
            # Update image
            optimizer.step()

            # Print interation
            self.print_inter_loss(i,step,n_iter,loss)
            # precondition
            init_image = self.precondition_metric(GB_radius)

        # remove hook
        handle.remove()

        # output synthesized image
        final_image = self.optimal_image[0].detach().numpy().copy()
        # if save_path is not None:
        #     img_out = ip.to_pil(final_image, True)
        #     img_out.save(pjoin(save_path, f'synthesized_image_iter{n_iter}.jpg'))
        return final_image

