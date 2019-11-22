# import some necessary packages
import numpy as np
import torch, copy, cv2
from os import remove
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate
from dnnbrain.dnn.core import Mask
from dnnbrain.dnn import models as db_models # Use eval to import DNN model


class Algorithm():

    """
    An abstract method to contain some necessary parameters to be used for
    calculating upsampling activation map / upsampling empirical feceptive field /
    occluder discrepancy map / occluder empirical receptive field.
    """

    def __init__(self, model, layer, channel):

        """
        Parameter:
        ---------
        model[str]: The name of DNN net.
                  You should open dnnbrain.dnn.models
                  to check if the DNN net is supported.
        layer[str]: The name of layer in DNN net.
        channel[int]: The channel of layer which you focus on.
        """

        self.model = eval('db_models.{}()'.format(model))
        self.layer = layer
        self.channel = channel
        self.dmask = Mask()
        self.dmask.set(self.layer, [self.channel, ])

    def set_layer(self, layer, channel):

        """
        Parameter:
        ---------
        layer[str]: The name of layer in DNN net.
        channel[int]: The channel of layer which you focus on.
        """

        self.layer = layer
        self.channel = channel
        self.dmask = Mask()
        self.dmask.set(self.layer, [self.channel, ])


class UpsamplingActivationMap(Algorithm):

    """
        A class to compute activation for each pixel from an image by upsampling
        activation map, with specific method assigned.
    """

    def set_params(self, ip_metric='bicubic', up_thres=95):

        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        ip_metric[str]: Algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' |
            'trilinear' | 'area' |. Default: 'bicubic'
        up_thres[int]: The threshold to filter the feature map,
                    which you should assign between 0-99.
        """

        self.ip_metric = ip_metric
        self.up_thres = up_thres

    def compute(self, image):

        """
        The method do real computation for pixel activation based on feature mapping upsampling.

        Parameter:
        ---------
        image[np-array]: a W x H x 3 Numpy Array.
        """

        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        self.img = np.einsum('abc->cab', img)[np.newaxis, :]
        self.act = self.model.compute_activation(self.img, self.dmask).get(self.layer).squeeze()
        fm = torch.from_numpy(self.act)[np.newaxis, np.newaxis, ...]
        ip = interpolate(fm, size=self.img.shape[2:4], mode=self.ip_metric, align_corners=True)
        up_raw_map = np.squeeze(np.asarray(ip))
        up_raw_map = copy.deepcopy(up_raw_map)
        up_raw_map[up_raw_map < np.percentile(up_raw_map, self.up_thres)] = 0
        up_map = up_raw_map / np.max(up_raw_map)
        return up_map


class UpsamplingEmpiricalReceptiveField(Algorithm):

    """
    A class to estimate empiral receptive field of a DNN model.
    """

    def set_params(self, rf_thres=100):

        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        rf_thres[int]: The threshold to filter the synthesized
                      receptive field, which you should assign
                      between 0-99.
        """

        # self.activation_estimator.set_params(rf_thres)
        self.rf_thres = rf_thres

    def compute(self, up_maps):

        """
        Generate RF based on provided image and pixel activation estimator

        Parameter:
        ---------
        up_maps[np-array]: a N x 224 x 224 Numpy Array.
        """

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
        _, th = cv2.threshold(rf, self.rf_thres, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        rf = cv2.ellipse(rf, cv2.fitEllipse(contours[0]), (255, 255, 255), 1, 300)

        r = np.array(contours).squeeze()
        t = 0
        for i in np.unique(r[:, 0]):
            t = t + max(r[r[:, 0] == i, 1]) - min(r[r[:, 0] == i, 1])

        rf = cv2.putText(rf, 'RF\'s Size: ' + str(int(np.sqrt(t))), (0, 22),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return rf

    class OccluderDiscrepancyMap(Algorithm):

        """
        An class to compute activation for each pixel from an image
        using slide Occluder
        """

        def set_params(self, osize=(11, 11), stride=(2, 2), oc_metric='max'):
            """
            Set necessary parameters for the estimator

            Parameter:
            ---------
            osize[list]: The size of sliding window, which form should be [int, int].
            The window will start from top-left and slides from left to right,
            and then from top to bottom.

            stride[list]: The move step if sliding window, which form should be [int, int]
            The first element of stride is the step for rows, while the second element of
            stride is the step for column.

            oc_metric[str]: The way to measure discrepancy map.
            """

            self.osize = osize
            self.stride = stride
            self.oc_metric = oc_metric

        def compute(self, image):

            """
            Please implement the sliding occluder algothrim for discrepancy map.
            """

            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
            img = np.einsum('abc->cab', img)[np.newaxis, :]

            widenum = int((img.shape[2] - self.osize[0]) / self.stride[0] + 1)
            heightnum = int((img.shape[3] - self.osize[1]) / self.stride[0] + 1)

            dmap = np.zeros((widenum, heightnum))
            dmap0 = np.max(self.model.compute_activation(img, self.dmask).get(self.channel))

            r = 1

            for i in range(0, widenum):
                for j in range(0, heightnum):
                    tmpoc = copy.deepcopy(img)
                    tmpzeros = np.zeros((self.osize[0], self.osize[1], 3))
                    tmpoc[self.stride[0] * i:self.stride[0] * i + self.osize[0],
                          self.stride[1] * j:self.stride[1] * j + self.osize[1], :] = tmpzeros
                    tmpoc = np.einsum('abc->cab', tmpoc)[np.newaxis, :]
                    tmpmax = np.max(self.model.compute_activation(tmpoc, self.dmask).get(self.channel))
                    dmap[i, j] = dmap0 - tmpmax

                    print(r, 'in', widenum * heightnum,
                          'finished. Discrepancy: %.1f' % abs(dmap[i, j]))
                    r = r + 1

            return dmap

    class OccluderEmpiricalReceptiveField(Algorithm):
        """
        A class to estimate empiral receptive field of a DNN model
        """

        def set_params(self, rf_thres=100):

            """
            Set necessary parameters for upsampling estimator.

            Parameter:
            ---------
            rf_thres[int]: The threshold to filter the synthesized
                          receptive field, which you should assign
                          between 0-99.
            """

            # self.activation_estimator.set_params(rf_thres)
            self.rf_thres = rf_thres

        def compute(self, oc_maps):

            """
            Generate RF based on provided image and pixel activation estimator

            Parameter:
            ---------
            oc_maps[np-array]: a N x 224 x 224 Numpy Array.
            """

            sum_act = np.zeros([oc_maps.shape[0], 224 * 2 - 1, 224 * 2 - 1])
            for i in range(oc_maps.shape[0]):
                cx = int(np.mean(np.where(oc_maps[i, :, :] == np.max(oc_maps[i, :, :]))[0]))
                cy = int(np.mean(np.where(oc_maps[i, :, :] == np.max(oc_maps[i, :, :]))[1]))
                sum_act[i, 223 - cx:447 - cx, 223 - cy:447 - cy] = oc_maps[i, :, :]
            sum_act = np.sum(sum_act, 0)[112:336, 112:336]
            plt.imsave('tmp.png', sum_act, cmap='gray')
            rf = cv2.imread('tmp.png', cv2.IMREAD_GRAYSCALE)
            rf = cv2.medianBlur(rf, 31)
            _, th = cv2.threshold(rf, self.rf_thres, 255, cv2.THRESH_BINARY)
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
