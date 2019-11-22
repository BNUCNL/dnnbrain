# import some necessary packages
import numpy as np
import torch, copy, cv2
from os import remove
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate
from dnnbrain.dnn.core import Algorithm

class UpsamplingActivationMapping():
    """
        A class to compute activation for each pixel from an image by upsampling
        activation map, with specific method assigned.
    """
    
    def __init__(self, interp_meth='bicubic', interp_threshold=95):

        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        interp_meth[str]: Algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' |
            'trilinear' | 'area' |. Default: 'bicubic'
        up_thres[int]: The threshold to filter the feature map,
                    which you should assign between 0-99.
        """

        self.interp_meth = interp_meth
        self.interp_threshold = interp_threshold

    def set_params(self, interp_meth='bicubic', interp_threshold=95):

        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        interp_meth[str]: Algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' |
            'trilinear' | 'area' |. Default: 'bicubic'
        up_thres[int]: The threshold to filter the feature map,
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

        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        self.img = np.einsum('abc->cab', img)[np.newaxis, :]
        self.act = self.model.compute_activation(self.img, self.mask).get(self.layer).squeeze()
        fm = torch.from_numpy(self.act)[np.newaxis, np.newaxis, ...]
        ip = interpolate(fm, size=self.img.shape[2:4], mode=self.interp_meth, align_corners=True)
        up_raw_map = np.squeeze(np.asarray(ip))
        up_raw_map = copy.deepcopy(up_raw_map)
        up_raw_map[up_raw_map < np.percentile(up_raw_map, self.interp_threshold)] = 0
        up_map = up_raw_map / np.max(up_raw_map)
        return up_map


class OccluderDiscrepancyMapping():
    """
    An class to compute activation for each pixel from an image
    using slide Occluder
    """

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

        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = np.einsum('abc->cab', img)[np.newaxis, :]

        widenum = int((img.shape[2] - self.window[0]) / self.stride[0] + 1)
        heightnum = int((img.shape[3] - self.window[1]) / self.stride[0] + 1)

        dmap = np.zeros((widenum, heightnum))
        dmap0 = np.max(self.model.compute_activation(img, self.mask).get(self.channel))

        r = 1

        for i in range(0, widenum):
            for j in range(0, heightnum):
                tmpoc = copy.deepcopy(img)
                tmpzeros = np.zeros((self.window[0], self.window[1], 3))
                tmpoc[self.stride[0] * i:self.stride[0] * i + self.window[0],
                      self.stride[1] * j:self.stride[1] * j + self.window[1], :] = tmpzeros
                tmpoc = np.einsum('abc->cab', tmpoc)[np.newaxis, :]
                tmpmax = np.max(self.model.compute_activation(tmpoc, self.mask).get(self.channel))
                dmap[i, j] = dmap0 - tmpmax

                print(r, 'in', widenum * heightnum,
                      'finished. Discrepancy: %.1f' % abs(dmap[i, j]))
                r = r + 1

        return dmap

class EmpiricalReceptiveField(Algorithm):
    """
    A class to estimate empiral receptive field of a DNN model.
    """
    def __init__(self, dnn, layer=None, channel=None):
        super(EmpiricalReceptiveField, self).__init__(dnn, layer, channel)
        self.mapping = None
       
    def set_params(self, threshold=0.95):

        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        threshold[int]: The threshold to filter the synthesized
                      receptive field, which you should assign
                      between 0-1.
        """
        self.threshold = threshold

    def upsampling_mapping(self, image, interp_meth='bicubic', interp_threshold=95):
        """
        Compute activation for each pixel from an image by upsampling
        activation map, with specific method assigned. 
        """
        self.mapping = UpsamplingActivationMappingz(interp_meth, interp_threshold)
        self.activation_map=self.maping(image)

    def occluder_mapping(self, image, window=(11, 11), stride=(2, 2), metric='max'):
        """
        Compute activation for each pixel from an image using sliding occluder window
        """
        self.mapping = OccluderDiscrepancyMapping(window, stride=(2, 2), metric)
        self.activation_map=self.maping(image)
      
    def generate_rf(self):

        """
        Compute ERF  on provided image for target layer and channel.
        
        Note: before call this method, you should call xx_mapping method to 
        derive activation map in the image space. 
        """
        if self.activation_map is None: 
            raise AssertionError('Please first call upsampling_mapping or occluder_mappin to '
                                 'map activiton to image space')

        sum_act = np.zeros([up_maps.shape[0], 224 * 2 - 1, 224 * 2 - 1])
        for i in range(up_maps.shape[0]):
            cx = int(np.mean(np.where(up_maps[i, :, :] == np.max(up_maps[i, :, :]))[0]))
            cy = int(np.mean(np.where(up_maps[i, :, :] == np.max(up_maps[i, :, :]))[1]))
            sum_act[i, 223 - cx:447 - cx, 223 - cy:447 - cy] = up_maps[i, :, :]

        sum_act = np.sum(sum_act, 0)[112:336, 112:336]
        plt.imsave('tmp.png', sum_act, cmap='gray')
        rf = cv2.imread('tmp.png', cv2.IMREAD_GRAYSCALE)
        remove('tmp.png')
        rf = cv2.medianBlur(rf, 31)
        _, th = cv2.threshold(rf, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        rf = cv2.ellipse(rf, cv2.fitEllipse(contours[0]), (255, 255, 255), 1, 300)

        r = np.array(contours).squeeze()
        t = 0
        for i in np.unique(r[:, 0]):
            t = t + max(r[r[:, 0] == i, 1]) - min(r[r[:, 0] == i, 1])

        rf = cv2.putText(rf, 'RF\'s Size: ' + str(int(np.sqrt(t))), (0, 22),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return rf

class TheoreticalReceptiveField(Algorithm):
    """
    A class to count theoretical receptive field. Noted that now only AlexNet,
    Vgg16, Vgg19 are supported (all these net are linear structure).
    """
    def compute(self):
        if self.model.__class__.__name__ == 'AlexNet':
            self.net_struct = {}
            self.net_struct['net'] = [[11, 4, 0], [3, 2, 0], [5, 1, 2], [3, 2, 0],
                                      [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0]]
            self.net_struct['name'] = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3',
                                       'conv4', 'conv5', 'pool5']

        if self.dnn.__class__.__name__ == 'Vgg11':
            self.net_struct = {}
            self.net_struct['net'] = [[3, 1, 1], [2, 2, 0], [3, 1, 1], [2, 2, 0],
                                      [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [2, 2, 0]]
            self.net_struct['name'] = ['conv1', 'pool1', 'conv2', 'pool2',
                                       'conv3_1', 'conv3_2', 'pool3', 'conv4_1',
                                       'conv4_2', 'pool4', 'conv5_1', 'conv5_2',
                                       'pool5']

        if self.dnn.__class__.__name__ == 'Vgg16':
            self.net_struct['net'] = [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0]]
            self.net_struct['name'] = ['conv1_1', 'conv1_2', 'pool1', 'conv2_1',
                                       'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
                                       'conv3_3', 'pool3', 'conv4_1', 'conv4_2',
                                       'conv4_3', 'pool4', 'conv5_1', 'conv5_2',
                                       'conv5_3', 'pool5']

        if self.dnn.__class__.__name__ == 'Vgg19':
            self.net_struct['net'] = [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1],
                                      [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1],
                                      [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0],
                                      [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                      [2, 2, 0]]
            self.net_struct['name'] = ['conv1_1', 'conv1_2', 'pool1', 'conv2_1',
                                       'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
                                       'conv3_3', 'conv3_4', 'pool3', 'conv4_1',
                                       'conv4_2', 'conv4_3', 'conv4_4', 'pool4',
                                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                                       'pool5']
        rf_size = 1
        for layer in reversed(range(self.net_struct['name'].index(self.layer) + 1)):
            fsize, stride, padding = self.net_struct['net'][layer]
            rf_size = ((rf_size - 1) * stride) + fsize
        return rf_size
    # we better return a RF rather than RF size
