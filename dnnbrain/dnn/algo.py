import abc
import torch


class SaliencyImage(abc.ABC):
    """
    An Abstract Base Classes class to define interfaces for gradient back propagation
    """

    def __init__(self, dnn, first_layer):
        """
        Parameter:
        ---------
        dnn[DNN]: dnnbrain's DNN object
        first_layer[str]: layer name of the first layer
        """
        # about DNN
        self.dnn = dnn
        self.dnn.eval()

        # about layer location
        self.first_layer = first_layer
        self.target_layer = None
        self.chn_idx = None

        # about hook
        self.activation = []
        self.gradient = None
        self.hook_handles = []

    def set(self, layer, channel):
        """
        Set the target

        Parameters:
        ----------
        layer[str]: layer name
        channel[int]: channel number
        """
        self.target_layer = layer
        self.chn_idx = channel - 1

    @abc.abstractmethod
    def register_hooks(self):
        """
        Define register hook and register them to specific layer and channel.
        As this a abstract method, it is needed to be override in every subclass
        """

    def visualize(self, image):
        """
        Compute gradient corresponding to the target on the image with back propagation algorithm

        Parameter:
        ---------
        image[Tensor]: an input of the model, with shape as (1, n_chn, n_height, n_width)

        Return:
        ------
        gradient[ndarray]: the input's gradients corresponding to the target activation
            with shape as (n_chn, n_height, n_width)
        """
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("The input data must be a tensor with shape as "
                             "(1, n_chn, n_height, n_width)")

        self.register_hooks()
        # forward
        image.requires_grad_(True)
        self.dnn(image)
        # zero grads
        self.dnn.model.zero_grad()
        # backward
        self.activation.pop().backward()
        # tensor to ndarray
        # [0] to get rid of the first dimension (1, n_chn, n_height, n_weight)
        gradient = self.gradient.data.numpy()[0]
        self.gradient = None

        # remove hooks
        for hook_handle in self.hook_handles:
            hook_handle.remove()

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
            self.activation.append(torch.mean(feat_out[0, self.chn_idx]))

        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_in[0]

        # register forward hook to the target layer
        trg_module = self.dnn.layer2module(self.target_layer)
        self.hook_handles.append(trg_module.register_forward_hook(forward_hook))

        # register backward to the first layer
        first_module = self.dnn.layer2module(self.first_layer)
        self.hook_handles.append(first_module.register_backward_hook(backward_hook))
