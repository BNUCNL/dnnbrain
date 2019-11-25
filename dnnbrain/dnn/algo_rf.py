# import some necessary packages
import math
import numpy as np
from os import remove
import torch.nn as nn
import torch, copy, cv2
from collections import namedtuple
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate
from dnnbrain.dnn.core import Mask, Algorithm
from dnnbrain.dnn import models as db_models # Use eval to import model model


class UpsamplingActivationMapping(Algorithm):

    """
        A class to compute activation for each pixel from an image by upsampling
        activation map, with specific method assigned.
    """

    def __init__(self, model, layer, channel, interp_meth='bicubic', interp_threshold=0.68):

        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        interp_meth[str]: Algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' |
            'trilinear' | 'area' |. Default: 'bicubic'
        interp_threshold[int]: The threshold to filter the feature map,
                               which you should assign between 0-99.
        """

        super(UpsamplingActivationMapping, self).__init__(model, layer, channel)
        self.mask = Mask()
        self.mask.set(self.layer, [self.channel, ])
        self.interp_meth = interp_meth
        self.interp_threshold = interp_threshold

    def set_params(self, interp_meth='bicubic', interp_threshold=0.68):

        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        interp_meth[str]: Algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' |
            'trilinear' | 'area' |. Default: 'bicubic'
        interp_threshold[int]: The threshold to filter the feature map,
                    which you should assign between 0-99.
        """

        self.interp_meth = interp_meth
        self.interp_threshold = interp_threshold

    def compute(self, image):

        """
        The method do real computation for pixel activation based on feature mapping upsampling.

        Parameter:
        ---------
        image[np-array]: a W x H x 3 Numpy Array.
        """

        croped_img = cv2.resize(image, self.model.img_size, interpolation=cv2.INTER_CUBIC)
        self.croped_img = croped_img.transpose(2, 0, 1)[np.newaxis, :]
        self.img_act = self.model.compute_activation(self.croped_img, self.mask).get(self.layer).squeeze()
        self.img_act = torch.from_numpy(self.img_act)[np.newaxis, np.newaxis, ...]
        self.img_act = interpolate(self.img_act, size=self.croped_img.shape[2:4],
                                   mode=self.interp_meth, align_corners=True)
        self.img_act = np.squeeze(np.asarray(self.img_act))
        thresed_img_act = copy.deepcopy(self.img_act)

        thresed_img_act[thresed_img_act < np.percentile(thresed_img_act, self.interp_threshold * 100)] = 0
        thresed_img_act = thresed_img_act / np.max(thresed_img_act)
        self.thresed_img_act = thresed_img_act
        return self.thresed_img_act


class OccluderDiscrepancyMapping(Algorithm):

    """
    An class to compute activation for each pixel from an image
    using slide Occluder
    """

    def __init__(self, model, layer, channel, window=(11, 11), stride=(2, 2), metric='max'):
        super(UpsamplingActivationMapping, self).__init__(model, layer, channel)
        self.window = window
        self.stride = stride
        self.metric = metric

    def set_params(self, window=(11, 11), stride=(2, 2), metric='max'):

        """
        Set necessary parameters for the estimator

        Parameter:
        ---------
        window[list]: The size of sliding window, which form should be [int, int].
        The window will start from top-left and slides from left to right,
        and then from top to bottom.

        stride[list]: The move step if sliding window, which form should be [int, int]
        The first element of stride is the step for rows, while the second element of
        stride is the step for column.

        metric[str]: The metric to measure how feature map change, max or mean.
        """

        self.window = window
        self.stride = stride
        self.metric = metric

    def compute(self, image):

        """
        Please implement the sliding occluder algothrim for discrepancy map.
        """

        self.croped_img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        self.croped_img = self.croped_img.transpose(2, 0, 1)[np.newaxis, :]

        widenum = int((self.croped_img.shape[2] - self.window[0]) / self.stride[0] + 1)
        heightnum = int((self.croped_img.shape[3] - self.window[1]) / self.stride[0] + 1)

        dmap = np.zeros((widenum, heightnum))
        dmap0 = np.max(self.model.compute_activation(self.croped_img, self.mask).get(self.channel))

        r = 1

        for i in range(0, widenum):
            for j in range(0, heightnum):
                tmpoc = copy.deepcopy(self.croped_img)
                tmpoc[self.stride[0] * i:self.stride[0] * i + self.window[0],
                      self.stride[1] * j:self.stride[1] * j + self.window[1], :] = 0
                tmpoc = tmpoc.transpose(2, 0, 1)[np.newaxis, :]
                tmpmax = np.max(self.model.compute_activation(tmpoc, self.mask).get(self.channel))
                dmap[i, j] = dmap0 - tmpmax

                print(r, 'in', widenum * heightnum,
                      'finished. Discrepancy: %.1f' % abs(dmap[i, j]))
                r = r + 1

        return dmap


class EmpiricalReceptiveField():

    """
    A class to estimate empiral receptive field of a DNN model.
    """

    def __init__(self, model=None, layer=None, channel=None, threshold=0.3921):

        """
        Parameter:
        ---------
        threshold[int]: The threshold to filter the synthesized
                      receptive field, which you should assign
                      between 0-1.
        """

        super(EmpiricalReceptiveField, self).__init__(model, layer, channel)
        self.model = model
        self.threshold = threshold

    def set_params(self, threshold=0.3921):

        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        threshold[int]: The threshold to filter the synthesized
                      receptive field, which you should assign
                      between 0-1.
        """

        self.threshold = threshold

    def generate_rf(self, all_thresed_act):

        """
        Compute RF on provided image for target layer and channel.

        Parameter:
        ---------
        all_thresed_act[n x w x h]: The threshold to filter the synthesized
                      receptive field, which you should assign
                      between 0-1.

        """

        self.all_thresed_act = all_thresed_act
        sum_act = np.zeros([self.all_thresed_act.shape[0], self.model.img_size[0] * 2 - 1,
                            self.model.img_size[1] * 2 - 1])
        for pics_layer in range(self.all_thresed_act.shape[0]):
            cx = int(np.mean(np.where(self.all_thresed_act[pics_layer, :, :] ==
                                      np.max(self.all_thresed_act[pics_layer, :, :]))[0]))
            cy = int(np.mean(np.where(self.all_thresed_act[pics_layer, :, :] ==
                                      np.max(self.all_thresed_act[pics_layer, :, :]))[1]))
            sum_act[pics_layer, self.model.img_size[0] - 1 - cx:2 * self.model.img_size[0] - 1 - cx,
                    self.model.img_size[1] - 1 - cy:2 * self.model.img_size[1] - 1 - cy] = self.all_thresed_act[pics_layer, :, :]

        sum_act = np.sum(sum_act, 0)[int(self.model.img_size[0] / 2):int(self.model.img_size[0] * 3 / 2),
                                     int(self.model.img_size[1] / 2):int(self.model.img_size[1] * 3 / 2)]
        plt.imsave('tmp.png', sum_act, cmap='gray')
        rf = cv2.imread('tmp.png', cv2.IMREAD_GRAYSCALE)
        remove('tmp.png')
        rf = cv2.medianBlur(rf, 31)
        _, th = cv2.threshold(rf, self.threshold * 255, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # rf = cv2.ellipse(rf, cv2.fitEllipse(contours[0]), (255, 255, 255), 1, 300)
        rf_contour = np.array(contours).squeeze()
        rf_area = 0
        for i in np.unique(rf_contour[:, 0]):
            rf_area = rf_area + max(rf_contour[rf_contour[:, 0] == i, 1]) - min(rf_contour[rf_contour[:, 0] == i, 1])
        return np.sqrt(rf_area)


Size = namedtuple('Size', ('w', 'h'))
Vector = namedtuple('Vector', ('x', 'y'))


class TheoreticalReceptiveField(namedtuple('TheoreticalReceptiveField', ('offset', 'stride', 'rfsize', 'outputsize', 'inputsize'))):

    """
    Contains information of a network's receptive fields (RF).
    The RF size, stride and offset can be accessed directly,
    or used to calculate the coordinates of RF rectangles using
    the convenience methods.
    """

    def left(self):
        """
        Return left (x) coordinates of the receptive fields.
        """
        return torch.arange(float(self.outputsize.w)) * self.stride.x + self.offset.x

    def top(self):
        """
        Return top (y) coordinates of the receptive fields.
        """
        return torch.arange(float(self.outputsize.h)) * self.stride.y + self.offset.y

    def hcenter(self):
        """
        Return center (x) coordinates of the receptive fields.
        """
        return self.left() + self.rfsize.w / 2

    def vcenter(self):
        """
        Return center (y) coordinates of the receptive fields.
        """
        return self.top() + self.rfsize.h / 2

    def right(self):
        """
        Return right (x) coordinates of the receptive fields.
        """
        return self.left() + self.rfsize.w

    def bottom(self):
        """
        Return bottom (y) coordinates of the receptive fields.
        """
        return self.top() + self.rfsize.h

    def rects(self):
        """
        Return a list of rectangles representing the receptive fields of all output elements. Each rectangle is a tuple (x, y, width, height).
        """
        return [(x, y, self.rfsize.w, self.rfsize.h) for x in self.left().numpy() for y in self.top().numpy()]

    def show(self, image=None, axes=None, show=True):
        """
        Visualize receptive fields.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if image is None:
            # create a checkerboard image for the background
            xs = torch.arange(self.inputsize.w).unsqueeze(1)
            ys = torch.arange(self.inputsize.h).unsqueeze(0)
            image = (xs.remainder(8) >= 4) ^ (ys.remainder(8) >= 4)
            image = image * 128 + 64

        if axes is None:
            (fig, axes) = plt.subplots(1)

        # convert image to numpy and show it
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(-1, -2)
        axes.imshow(image, cmap='gray', vmin=0, vmax=255)

        rect_density = self.stride.x * self.stride.y / (self.rfsize.w * self.rfsize.h)
        rects = self.rects()

        for (index, (x, y, w, h)) in enumerate(rects):    # iterate RFs
            # show center marker
            marker, = axes.plot(x + w / 2, y + w / 2, marker='x')

            # show rectangle with some probability, since it's too dense.
            # also, always show the first and last rectangles for reference.
            if index == 0 or index == len(rects) - 1 or torch.rand(1).item() < rect_density:
                axes.add_patch(patches.Rectangle((x, y), w, h, facecolor=marker.get_color(), edgecolor='none', alpha=0.5))

        # set axis limits correctly
        axes.set_xlim(self.left().min().item(), self.right().max().item())
        axes.set_ylim(self.top().min().item(), self.bottom().max().item())
        axes.invert_yaxis()

        return plt.show()


(x_dim, y_dim) = (-1, -2)    # indexes of spatial dimensions in tensors


def receptivefield(model, input_shape, device='cpu'):
    """
    Computes the receptive fields for the given network (nn.Module) and input shape, given as a tuple (images, channels, height, width).
    Returns a TheoreticalReceptiveField object.
    """

    if len(input_shape) < 4:
        raise ValueError('Input shape must be at least 4-dimensional (N x C x H x W).')

    hooks = []

    def insert_hook(module):
        if isinstance(module, (nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d)):
            hook = _passthrough_grad
            if isinstance(module, nn.MaxPool2d):
                hook = _maxpool_passthrough_grad
            hooks.append(module.register_backward_hook(hook))
    model.apply(insert_hook)

    mode = model.training
    model.eval()

    input = torch.ones(input_shape, requires_grad=True, device=device)
    output = model(input)

    if output.dim() < 4:
        raise ValueError('Network is fully connected (output should have at least 4 dimensions: N x C x H x W).')

    outputsize = Size(output.shape[x_dim], output.shape[y_dim])
    if outputsize.w < 2 and outputsize.h < 2:
        raise ValueError('Network output is too small along spatial dimensions (fully connected).')

    (x1, x2, y1, y2, pos) = _project_rf(input, output, return_pos=True)
    rfsize = Size(x2 - x1 + 1, y2 - y1 + 1)

    (x1o, _, _, _) = _project_rf(input, output, offset_x=1)
    (_, _, y1o, _) = _project_rf(input, output, offset_y=1)
    stride = Vector(x1o - x1, y1o - y1)

    if stride.x == 0 and stride.y == 0:
        raise ValueError('Input tensor is too small relative to network receptive field.')

    offset = Vector(x1 - pos[x_dim] * stride.x, y1 - pos[y_dim] * stride.y)

    for hook in hooks:
        hook.remove()
    model.train(mode)

    inputsize = Size(input_shape[x_dim], input_shape[y_dim])
    return TheoreticalReceptiveField(offset, stride, rfsize, outputsize, inputsize)


def _project_rf(input, output, offset_x=0, offset_y=0, return_pos=False):
    """
    Project one-hot output gradient, using back-propagation, and return its bounding box at the input.
    """

    pos = [0] * len(output.shape)
    pos[x_dim] = math.ceil(output.shape[x_dim] / 2) - 1 + offset_x
    pos[y_dim] = math.ceil(output.shape[y_dim] / 2) - 1 + offset_y

    out_grad = torch.zeros(output.shape)
    out_grad[tuple(pos)] = 1

    if input.grad is not None:
        input.grad.zero_()

    output.backward(gradient=out_grad, retain_graph=True)

    in_grad = input.grad[0, 0]
    is_inside_rf = (in_grad != 0.0)

    xs = is_inside_rf.any(dim=y_dim).nonzero()
    ys = is_inside_rf.any(dim=x_dim).nonzero()

    if xs.numel() == 0 or ys.numel() == 0:
        raise ValueError('Could not propagate gradient through network to determine receptive field.')

    bounds = (xs.min().item(), xs.max().item(), ys.min().item(), ys.max().item())
    if return_pos:
        return (*bounds, pos)
    return bounds


def _passthrough_grad(self, grad_input, grad_output):
    """Hook to bypass normal gradient computation (of first input only)."""
    if isinstance(grad_input, tuple) and len(grad_input) > 1:
        return (grad_output[0], *grad_input[1:])
    else:
        return grad_output


def _maxpool_passthrough_grad(self, grad_input, grad_output):
    """Hook to bypass normal gradient computation of nn.MaxPool2d."""
    assert isinstance(self, nn.MaxPool2d)
    if self.dilation != 1 and self.dilation != (1, 1):
        raise ValueError('Dilation != 1 in max pooling not supported.')

    with torch.enable_grad():
        input = torch.ones(grad_input[0].shape, requires_grad=True)
        output = nn.functional.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode)
        return torch.autograd.grad(output, input, grad_output[0])
