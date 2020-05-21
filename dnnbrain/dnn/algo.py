import abc
import cv2
import time
import copy
import torch
import numpy as np

from os import remove
from os.path import join as pjoin
from scipy.ndimage.filters import gaussian_filter
from torch.optim import Adam
from torch.nn.functional import interpolate
from matplotlib import pyplot as plt
from dnnbrain.dnn.core import Mask
from dnnbrain.dnn.base import ip, array_statistic
from skimage import filters, segmentation
from skimage.color import rgb2gray
from skimage.morphology import convex_hull_image, erosion, square


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
        for iter_idx in range(1, n_iter + 1):
            # prepare image
            image_noisy = image + image.normal_(0, sigma ** 2)
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
            self.activation = torch.mean(feat_out[0, from_chn - 1])

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
                 activ_metric='mean', regular_metric=None, precondition_metric=None, smooth_metric=None,
                 save_out_interval=False, print_inter_loss=False):
        """
        Parameters:
        ----------
        dnn[DNN]: DNNBrain DNN
        layer[str]: name of the layer where the algorithm performs on
        channel[int]: sequence number of the channel where the algorithm performs on
        activ_metric[str]: The metric method to summarize activation
        regular_metric[str]: The metric method of regularization
        precondition_metric[str]: The metric method of precondition
        smooth_metric[str]: the metric method of smoothing
        """
        super(SynthesisImage, self).__init__(dnn, layer, channel)
        self.set_metric(activ_metric, regular_metric, precondition_metric, smooth_metric)
        self.set_utiliz(save_out_interval, print_inter_loss)
        self.activ_loss = None
        self.optimal_image = None

        # loss recorder
        self.activ_losses = []
        self.regular_losses = []

        self.row =None
        self.column=None
        
    def set_metric(self, activ_metric, regular_metric,
                   precondition_metric, smooth_metric):
        """
        Set metric methods

        Parameter:
        ---------
        activ_metric[str]: The metric method to summarize activation
        regular_metric[str]: The metric method of regularization
        precondition_metric[str]: The metric method of preconditioning
        smooth_metric[str]: the metric method of smoothing
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
            
        # smooth metric setting
        if smooth_metric is None:
            self.smooth_metric = self._smooth_default
        elif smooth_metric == 'Fourier':
            self.smooth_metric = self._smooth_fourier
        else:
            raise AssertionError('Only Fourier Smooth is supported!')

    def set_utiliz(self, save_out_interval=False, print_inter_loss=False):
        '''
        Set print and save interval pics
        Parameters
        -----------
        save_out_interval[bool]
        print_inter_loss[bool]
        '''
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

    def _print_loss(self, i, step, n_iter, loss):
        if i % step == 0:
            print(f'Interation: {i}/{n_iter}; Loss: {loss}')

    def _print_close(self, i, step, n_iter, loss):
        pass

    def _save_out(self, currti, save_interval, save_path):
        if (currti + 1) % save_interval == 0 and save_path is not None:
            img_out = self.optimal_image[0].detach().numpy().copy()
            img_out = ip.to_pil(img_out, True)
            img_out.save(pjoin(save_path, f'synthesized_image_iter{currti + 1}.jpg'))
            print('Saved No.',currti + 1,'in iteration')
        elif save_path == None:
            raise AssertionError('Check save_out_interval parameters please! You must give save_interval & save_path!')

    def _close_save(self, currti, save_interval, save_path):
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

    def _smooth_default(self, factor):
        pass

    def _smooth_fourier(self, factor):
        """
        Tones down the optimal image gradient with 1/sqrt(f) filter in the Fourier domain.
        Equivalent to low-pass filtering in the spatial domain.
        
        Parameter:
        ---------
        factor[float]: parameters used in fourier transform
        """
        # initialize grad
        grad = self.optimal_image.grad
        # handle special situations
        if factor == 0:
            pass
        else:
            # get information of grad
            h, w = grad.size()[-2:]
            tw = np.minimum(np.arange(0, w), np.arange(w-1, -1, -1), dtype=np.float32) 
            th = np.minimum(np.arange(0, h), np.arange(h-1, -1, -1), dtype=np.float32)
            # filtering in the spatial domain
            t = 1 / np.maximum(1.0, (tw[None, :] ** 2 + th[:, None] ** 2) ** (factor))
            F = grad.new_tensor(t / t.mean()).unsqueeze(-1)
            pp = torch.rfft(grad.data, 2, onesided=False)
            # adjust the optimal_image grad after Fourier transform
            self.optimal_image.grad = torch.irfft(pp * F, 2, onesided=False)
    
    def mean_image(self):
        pass

    def center_bias(self):
        pass
        
    def register_hooks(self,unit=None):
        """
        Define register hook and register them to specific layer and channel.
        Parameter:
        ---------
        unit[tuple]: determine unit position, None means channel
        
        """
        layer, chn = self.get_layer()

        def forward_hook(module, feat_in, feat_out):
            if unit == None:
                self.activ_loss = - self.activ_metric(feat_out[0, chn - 1])
            else:
                if isinstance(unit,tuple) and len(unit)==2:
                    self.row,self.column = unit
                    row = int(self.row)
                    column = int(self.column)
                    self.activ_loss = - self.activ_metric(feat_out[0, chn - 1,row,column])  # single unit
                else:
                    raise AssertionError('Check unit must be 2-dimensinal tuple')
            self.activ_losses.append(self.activ_loss.item())

        # register forward hook to the target layer
        module = self.dnn.layer2module(layer)
        handle = module.register_forward_hook(forward_hook)

        return handle

    def synthesize(self, init_image = None, unit=None, lr = 0.1,
                    regular_lambda = 1, n_iter = 30, save_path = None,
                    save_interval = None, GB_radius = 0.875, factor = 0.5, step = 1):
        """
        Synthesize the image which maximally activates target layer and channel

        Parameter:
        ---------
        init_image[ndarray|Tensor|PIL.Image]: initialized image
        unit[tuple]:set target unit position
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
        factor[float]
        GB_radius[float]
        step[int]
        Return:
        ------
            [ndarray]: the synthesized image with shape as (n_chn, height, width)
        """
        # Hook the selected layer
        handle = self.register_hooks(unit)

        # prepare initialized image
        if init_image is None:
            # Generate a random image
            init_image = np.random.normal(loc=[0.485, 0.456, 0.406], scale=[0.229, 0.224, 0.225], 
                                          size=(*self.dnn.img_size, 3)).transpose(2,0,1)
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
            self.save_out_interval(i, save_interval, save_path)
            
            # Forward pass layer by layer until the target layer
            # to triger the hook funciton.
            self.dnn.model(self.optimal_image)
            # computer loss
            loss = self.activ_loss + regular_lambda * self.regular_metric()
            # zero gradients
            optimizer.zero_grad()
            # Backward
            loss.backward()
            #smoooth the gradient
            self.smooth_metric(factor)
            # Update image
            optimizer.step()
            
            # Print interation
            self.print_inter_loss(i, step, n_iter, loss)
            # precondition
            init_image = self.precondition_metric(GB_radius)
           
        # compute act_loss for one more time as the loss 
        # we computed in each iteration is for the previous pic
        self.dnn.model(self.optimal_image)
        self.regular_metric()   
                
        # remove hook
        handle.remove()

        # output synthesized image
        final_image = self.optimal_image[0].detach().numpy().copy()

        return final_image
    
    
class MaskedImage(Algorithm):
    '''
    Generate masked gray picture for images according to activation changes
    '''
    
    def __init__(self,dnn, layer=None, channel=None, unit=None, 
             stdev_size_thr=1.0,filter_sigma=1.0,target_reduction_ratio=0.9):
        """
        Parameters:
        ----------
        dnn[DNN]: DNNBrain DNN
        layer[str]: name of the layer where the algorithm performs on
        channel[int]: sequence number of the channel where the algorithm performs on
        initial_image[ndarray]: initial image waits for masking
        unit[tuple]: position of the target unit
        """
        super(MaskedImage, self).__init__(dnn, layer, channel)
        self.set_parameters(unit, stdev_size_thr, filter_sigma, target_reduction_ratio)
        self.activ = None
        self.masked_image = None
        self.activ_type = None
        self.row =None
        self.column=None
                
    def set_parameters(self, unit=None, stdev_size_thr=1.0,
                          filter_sigma=1.0, target_reduction_ratio=0.9):
        """
        Set parameters for mask
        
        Parameters
        ----------
        unit[tuple]: position of the target unit
        stdev_size_thr[float]: fraction of standard dev threshold for size of blobs,default 1.0
        filter_sigma[float]: sigma for final gaussian blur, default 1.0
        target_reduction_ratio[float]; reduction ratio to achieve for tightening the mask,default 0.9
        """            
        if isinstance(unit,tuple) and len(unit) == 2:    
            self.row,self.column = unit
            self.activ_type = 'unit'
        elif unit == None:
            self.activ_type = 'channel'
        else: 
            raise AssertionError('Check unit must be 2-dimentional tuple,like(27,27)')
        self.stdev_size_thr = stdev_size_thr
        self.filter_sigma = filter_sigma
        self.target_reduction_ratio = target_reduction_ratio
        
    def prepare_test(self,masked_image):
        '''
        transfer pic to tenssor for dnn activation
        
        Parameters:
        -----------
        masked_image [ndarray]: masked image waits for dnn activation
        
        returns:
        -----------
            [tensor] for dnn computation
        '''
        test_image = np.repeat(masked_image,3).reshape((224,224,3))
        test_image = test_image.transpose((2,0,1))
        test_image = ip.to_tensor(test_image).float()
        test_image = copy.deepcopy(test_image)
        test_image = test_image.unsqueeze(0)
        return test_image
    
    def register_hooks(self):
        """
        Define register hook and register them to specific layer and channel.
        """
        layer, chn = self.get_layer()

        def forward_hook(module, feat_in, feat_out):
            if self.activ_type=='channel':
                self.activ = torch.mean(feat_out[0, chn - 1])
            elif self.activ_type=='unit':
                row = int(self.row)
                column = int(self.column)
                self.activ = feat_out[0, chn - 1,row,column]  # single unit
            self.activ_trace.append(self.activ.item())

        # register forward hook to the target layer
        module = self.dnn.layer2module(layer)
        handle = module.register_forward_hook(forward_hook)

        return handle
             
    def put_mask(self, initial_image, maxiteration=100):
        '''
        Put mask on image
       
        Parameter:
        --------
        initial_image[ndarray]: initial image waits for masking
        maxiteration[int]: the max number of iterations to sto
        
        Return
        -------
        masked_image[ndarray]: the masked image with shape as (n_chn, height, width)
        '''
        if isinstance(initial_image,np.ndarray):
            if len(initial_image.shape) in [2,3]:
                img = initial_image
            else: 
                raise AssertionError('Check initial_image, only two or three dimentions can be set!')
        else:
            raise AssertionError('Check initial_image to be np.ndarray')
        #define hooks for recording act_loss
        self.activ_trace = [] 
        handle = self.register_hooks()
        
        #transpose axis
        if len(img.shape) == 3 and img.shape[0] == 3:        
            img = img.transpose((1,2,0))
        
        #degrade dimension
        img = rgb2gray(img)
        
        #compute the threshold of pixel contrast
        delta = img - img.mean()
        fluc = np.abs(delta)
        thr = np.std(fluc) * self.stdev_size_thr

        # original mask
        mask = convex_hull_image((fluc > thr).astype(float))
        fm = gaussian_filter(mask.astype(float), sigma=self.filter_sigma)
        masked_img = fm * img + (1 - fm) * img.mean()

        #prepare test img and get base acivation
        test_image = self.prepare_test(masked_img)
        self.dnn.model(test_image)
        activation = base_line = self.activ.detach().numpy()

        print('Baseline:', base_line)
        count = 0
        
        #START
        while (activation > base_line * self.target_reduction_ratio):
            mask = erosion(mask, square(3))
            
            #print('mask',mask)
            fm = gaussian_filter(mask.astype(float), sigma=self.filter_sigma)
            masked_img = fm * img + (1 - fm) * img.mean()
            test_image = self.prepare_test(masked_img)
            self.dnn.model(test_image)
            activation  = - self.activ_loss.detach().numpy()
            print('Activation:', activation)
            count += 1

            if  count > maxiteration:
                print('This has been going on for too long! - aborting')
                raise ValueError('The activation does not reduce for the given setting')
                break
        
        handle.remove()
        masked_image = test_image[0].detach().numpy()
        return  masked_image
    
        
class MinimalParcelImage(Algorithm):
    """
    A class to generate minimal image for target channels from a DNN model 
    """
    
    def __init__(self, dnn, layer=None, channel=None, activaiton_criterion='max', search_criterion='max'):
        """
        Parameter:
        ---------
        dnn[DNN]: dnnbrain's DNN object
        layer[str]: name of the layer where you focus on
        channel[int]: sequence number of the channel where you focus on
        activaiton_criterion[str]: the criterion of how to pooling activaiton
        search_criterion[str]: the criterion of how to search minimal image
        """
        super(MinimalParcelImage, self).__init__(dnn, layer, channel)
        self.set_params(activaiton_criterion, search_criterion)
        self.parcel = None
       
    def set_params(self, activaiton_criterion='max', search_criterion='max'):
        """
        Set parameter for searching minmal image
        
        Parameter:
        ---------
        activaiton_criterion[str]: the criterion of how to pooling activaiton, choices=(max, mean, median, L1, L2)
        search_criterion[str]: the criterion of how to search minimal image, choices=(max, fitting curve)
        """
        self.activaiton_criterion = activaiton_criterion
        self.search_criterion = search_criterion

    def _generate_decompose_parcel(self, image, segments):
        """
        Decompose image to multiple parcels using the given segments and
        put each parcel into a separated image with a black background
        
        Parameter:
        ---------
        image[ndarray]: shape (height,width,n_chn) 
        segments[ndarray]: shape (width, height).Integer mask indicating segment labels.
        
        Return:
        ---------
        parcel[ndarray]: shape (n_parcel,height,width,n_chn)
        """
        self.parcel = np.zeros((np.max(segments)+1,image.shape[0],image.shape[1],3),dtype=np.uint8)
        #generate parcel
        for label in np.unique(segments):
            self.parcel[label][segments == label] = image[segments == label]
        return self.parcel
        
    def felzenszwalb_decompose(self, image, scale=100, sigma=0.5, min_size=50):
        """
        Decompose image to multiple parcels using felzenszwalb method and
        put each parcel into a separated image with a black background
        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        
        Return:
        ---------
        parcel[ndarray]: shape (n_parcel,height,width,n_chn)
        """
        #decompose image
        segments = segmentation.felzenszwalb(image, scale, sigma, min_size)
        #generate parcel
        self.parcel = self._generate_decompose_parcel(image, segments)
        return self.parcel

    def slic_decompose(self, image, n_segments=250, compactness=10, sigma=1):
        """
        Decompose image to multiple parcels using slic method and
        put each parcel into a separated image with a black background  
        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        meth[str]: method to decompose images
        
        Return:
        ---------
        parcel[ndarray]: shape (n_parcel,height,width,n_chn)
        """
        #decompose image
        segments = segmentation.slic(image, n_segments, compactness, sigma)
        #generate parcel
        self.parcel = self._generate_decompose_parcel(image, segments)
        return self.parcel

    def quickshift_decompose(self, image, kernel_size=3, max_dist=6, ratio=0.5):
        """
        Decompose image to multiple parcels using quickshift method and
        put each parcel into a separated image with a black background
        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        meth[str]: method to decompose images
        
        Return:
        ---------
        parcel[ndarray]: shape (n_parcel,height,width,n_chn)
        """
        #decompose image
        segments = segmentation.quickshift(image, kernel_size, max_dist, ratio)
        #generate parcel
        self.parcel = self._generate_decompose_parcel(image, segments)
        return self.parcel
    
    def sort_parcel(self, order='descending'):
        """
        sort the parcel according the activation of dnn. 
        
        Parameter:
        ---------
        order[str]: ascending or descending
        
        Return:
        ---------
        parcel[ndarray]: shape (n_parcel,height,width,n_chn) parcel after sorted
        """
        #change its shape(n_parcel,n_chn,height,width)
        parcel = self.parcel.transpose((0,3,1,2))
        #compute activation
        dnn_acts = self.dnn.compute_activation(parcel, self.mask).pool(self.activaiton_criterion).get(self.mask.layers[0])
        act_all = dnn_acts.flatten()
        #sort the activation in order
        if order == 'descending':
            self.parcel = self.parcel[np.argsort(-act_all)]
        else:
            self.parcel = self.parcel[np.argsort(act_all)]
        return self.parcel
        
    def combine_parcel(self, indices):
        """
        combine the indexed parcel into a image
        
        Parameter:
        ---------
        indices[list|slice]: subscript indices
        
        Return:
        -----
        image_container[ndarray]: shape (n_chn,height,width)
        """
        #compose parcel correaspond with indices
        if isinstance(indices, (list,slice)):
            image_compose = np.sum(self.parcel[indices],axis=0)
        else:
            raise AssertionError('Only list and slice indices are supported')
        return image_compose
    
    def generate_minimal_image(self):
        """
        Generate minimal image. We first sort the parcel by the activiton and 
        then iterate to find the combination of the parcels which can maximally
        activate the target channel.
        
        Note: before call this method, you should call xx_decompose method to 
        decompose the image into parcels. 
        
        Return:
        ---------
        image_min[ndarray]: final minimal images in shape (height,width,n_chn)
        """
        if self.parcel is None: 
            raise AssertionError('Please run decompose method to '
                                 'decompose the image into parcels')
        # sort the image
        self.sort_parcel()
        # iterater combine image to get activation
        parcel_add = np.zeros((self.parcel.shape[0],self.parcel.shape[1],self.parcel.shape[2],3),dtype=np.uint8)
        for index in range(self.parcel.shape[0]):
            parcel_mix = self.combine_parcel(slice(index+1))
            parcel_add[index] = parcel_mix[np.newaxis,:,:,:]
        # change its shape(n_parcel,n_chn,height,width) to fit dnn_activation
        parcel_add = parcel_add.transpose((0,3,1,2))
        # get activation
        dnn_act = self.dnn.compute_activation(parcel_add, self.mask).pool(self.activaiton_criterion).get(self.mask.layers[0])
        act_add = dnn_act.flatten()
        # generate minmal image according to the search_criterion
        intere = 10
        if self.search_criterion == 'max':
            image_min = parcel_add[np.argmax(act_add[0:intere])]
            image_min = np.squeeze(image_min).transpose(1,2,0)
        else:
            pass
        return image_min


class MinimalComponentImage(Algorithm):
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """

    def set_params(self,  meth='pca', criterion='max'):
        """Set parameter for the estimator"""
        self.meth = meth
        self.criterion = criterion
        
    def pca_decompose(self):
        pass
    
    def ica_decompose(self):
        pass
    
    def sort_componet(self, order='descending'):
        """
        sort the component according the activation of dnn. 
        order[str]: ascending or descending
        """
        pass
    
    def combine_component(self, index):
        """combine the indexed component into a image"""
        pass 
    
    def generate_minimal_image(self):
        """
        Generate minimal image. We first sort the component by the activiton and 
        then iterate to find the combination of the components which can maximally
        activate the target channel.
        
        Note: before call this method, you should call xx_decompose method to 
        decompose the image into parcels. 
        
        Parameters:
        ---------
        stim[Stimulus]: stimulus
        
        Returns:
        ------
        """
        pass
    
        
class OccluderDiscrepancyMapping(Algorithm):
    """
    Slide a occluder window on an image, and calculate the change of
    the target channel's activation after each step.
    """

    def __init__(self, dnn, layer=None, channel=None, window=(11, 11), stride=(2, 2), metric='mean'):
        """
        Set necessary parameters.

        Parameters:
        ----------
        dnn[DNN]: DNNBrain's DNN object
        layer[str]: name of the layer that you focus on
        channel[int]: sequence number of the channel that you focus on (start with 1)
        window[tuple]: The size of sliding window - (width, height).
        stride[tuple]: The step length of sliding window - (width_step, height_step)
        metric[str]: max or mean
            The metric to summarize the target channel's activation
        """
        super(OccluderDiscrepancyMapping, self).__init__(dnn, layer, channel)
        self.set_params(window, stride, metric)
        
    def set_params(self, window, stride, metric):
        """
        Set parameter for occluder discrepancy mapping

        Parameters:
        ----------
        window[tuple]: The size of sliding window - (width, height).
        stride[tuple]: The step length of sliding window - (width_step, height_step)
        metric[str]: max or mean
            The metric to summarize the target channel's activation
        """        
        self.window = window
        self.stride = stride
        self.metric = metric

    def compute(self, image):
        """
        Compute discrepancy map of the image using a occluder window
        moving from top-left to bottom-right
        
        Parameter:
        ---------
        image[ndarray|Tensor|PIL.Image]: an original image
        
        Return:
        ---------
        discrepancy_map[ndarray]
        """
        # preprocess image
        image = ip.to_array(image)
        image = copy.deepcopy(image)[None, :]

        # initialize discrepancy map
        img_h, img_w = self.dnn.img_size
        win_w, win_h = self.window
        step_w, step_h = self.stride
        n_row = int((img_h - win_h) / step_h + 1)
        n_col = int((img_w - win_w) / step_w + 1)
        discrepancy_map = np.zeros((n_row, n_col))
        activ_ori = array_statistic(self.dnn.compute_activation(image, self.mask).get(self.mask.layers[0]),
                                    self.metric)

        # start computing by moving occluders
        for i in range(n_row):
            start = time.time()
            for j in range(n_col):
                occluded_img = copy.deepcopy(image)
                occluded_img[:, :, step_h*i:step_h*i+win_h, step_w*j:step_w*j+win_w] = 0
                activ_occ = array_statistic(self.dnn.compute_activation(
                    occluded_img, self.mask).get(self.mask.layers[0]), self.metric)
                discrepancy_map[i, j] = activ_ori - activ_occ
            print('Finished: row-{0}/{1}, cost {2} seconds'.format(
                i+1, n_row, time.time()-start))

        return discrepancy_map


class UpsamplingActivationMapping(Algorithm):
    """
    Resample the target channel's feature map to the input size
    after threshold.
    """

    def __init__(self, dnn, layer=None, channel=None, interp_meth='bicubic', interp_threshold=None):
        """
        Set necessary parameters.

        Parameters:
        ----------
        dnn[DNN]: dnnbrain's DNN object
        layer[str]: name of the layer where you focus on
        channel[int]: sequence number of the channel where you focus on
        interp_meth[str]: Algorithm used for resampling are
                          'nearest'  | 'bilinear' | 'bicubic' | 'area'
                          Default: 'bicubic'
        interp_threshold[int]: value is in [0, 1]
            The threshold used to filter the map after resampling.
            For example, if the threshold is 0.58, it means clip the feature map
            with the min as the minimum of the top 42% activation.
        """
        super(UpsamplingActivationMapping, self).__init__(dnn, layer, channel)
        self.set_params(interp_meth, interp_threshold)

    def set_params(self, interp_meth, interp_threshold):
        """
        Set necessary parameters.

        Parameters:
        ----------
        interp_meth[str]: Algorithm used for resampling are
                          'nearest'  | 'bilinear' | 'bicubic' | 'area'
                          Default: 'bicubic'
        interp_threshold[int]: value is in [0, 1]
            The threshold used to filter the map after resampling.
            For example, if the threshold is 0.58, it means clip the feature map
            with the min as the minimum of the top 42% activation.
        """
        self.interp_meth = interp_meth
        self.interp_threshold = interp_threshold

    def compute(self, image):
        """
        Resample the channel's feature map to input size

        Parameter:
        ---------
        image[ndarray|Tensor|PIL.Image]: an input image
        
        Return:
        ------
        img_act[ndarray]: shape=(height, width)
            image after resampling
        """
        # preprocess image
        image = ip.to_array(image)[None, :]

        # compute activation
        img_act = self.dnn.compute_activation(image, self.mask).get(self.mask.layers[0])

        # resample
        img_act = torch.from_numpy(img_act)
        img_act = interpolate(img_act, size=self.dnn.img_size, mode=self.interp_meth)
        img_act = np.squeeze(np.asarray(img_act))

        # threshold
        if self.interp_threshold is not None:
            thr = np.percentile(img_act, self.interp_threshold * 100)
            img_act = np.clip(img_act, thr, None)

        return img_act


class EmpiricalReceptiveField(Algorithm):
    """
    A Class to Estimate Empirical Receptive Field (RF) of a DNN Model.
    """

    def __init__(self, dnn, layer=None, channel=None, threshold=0.3921):
        """
        Parameter:
        ---------
        dnn[DNN]: dnnbrain's DNN object
        layer[str]: name of the layer where you focus on
        channel[int]: sequence number of the channel where you focus on
        threshold[int]: The threshold to filter the synthesized
                      receptive field, which you should assign
                      between 0 - 1.
        """
        super(EmpiricalReceptiveField, self).__init__(dnn, layer, channel)
        self.set_params(threshold)

    def set_params(self, threshold=0.3921):
        """
        Set necessary parameters for upsampling estimator.

        Parameter:
        ---------
        interp_meth[str]: Algorithm used for upsampling are
                          'nearest'   | 'linear' | 'bilinear' | 'bicubic' |
                          'trilinear' | 'area'   | 'bicubic' (Default)
        interp_threshold[int]: The threshold to filter the feature map, num between 0 - 99.
        """
        self.threshold = threshold

    def generate_rf(self, all_thresed_act):
        """
        Compute RF on Given Image for Target Layer and Channel

        Parameter:
        ---------
        all_thresed_act[ndarray]: shape must be (n_chn, dnn.img_size)
        
        Return:
        ---------
        empirical_rf_size[np.float64] : empirical rf size of specific image     
        """
        #init variables
        self.all_thresed_act = all_thresed_act
        sum_act = np.zeros([self.all_thresed_act.shape[0],
                            self.dnn.img_size[0] * 2 - 1, self.dnn.img_size[1] * 2 - 1])
        #compute act of image
        for current_layer in range(self.all_thresed_act.shape[0]):

            cx = int(np.mean(np.where(self.all_thresed_act[current_layer, :, :] ==
                                      np.max(self.all_thresed_act[current_layer, :, :]))[0]))

            cy = int(np.mean(np.where(self.all_thresed_act[current_layer, :, :] ==
                                      np.max(self.all_thresed_act[current_layer, :, :]))[1]))

            sum_act[current_layer,
                    self.dnn.img_size[0] - 1 - cx:2 * self.dnn.img_size[0] - 1 - cx,
                    self.dnn.img_size[1] - 1 - cy:2 * self.dnn.img_size[1] - 1 - cy] = \
                self.all_thresed_act[current_layer, :, :]

        sum_act = np.sum(sum_act, 0)[int(self.dnn.img_size[0] / 2):int(self.dnn.img_size[0] * 3 / 2),
                                     int(self.dnn.img_size[1] / 2):int(self.dnn.img_size[1] * 3 / 2)]
        #get region of receptive field
        plt.imsave('tmp.png', sum_act, cmap='gray')
        rf = cv2.imread('tmp.png', cv2.IMREAD_GRAYSCALE)
        remove('tmp.png')
        rf = cv2.medianBlur(rf, 31)
        _, th = cv2.threshold(rf, self.threshold * 255, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        rf_contour = np.vstack((np.array(contours)[0].squeeze(1),np.array(contours)[1].squeeze(1)))
        empirical_rf_area = 0
        #compute size of rf
        for i in np.unique(rf_contour[:, 0]):
            empirical_rf_area = empirical_rf_area + max(rf_contour[rf_contour[:, 0] == i, 1]) - \
                min(rf_contour[rf_contour[:, 0] == i, 1])
        empirical_rf_size = np.sqrt(empirical_rf_area)
        return empirical_rf_size


class TheoreticalReceptiveField(Algorithm):
    """
    A Class to Count Theoretical Receptive Field.
    Note: Currently only AlexNet, Vgg16, Vgg19 are supported.
    (All these net are linear structure.)
    """
    
    def __init__(self, dnn, layer=None, channel=None):
        """
        Parameter:
        ---------
        dnn[DNN]: dnnbrain's DNN object
        layer[str]: name of the layer where you focus on
        channel[int]: sequence number of the channel where you focus on
        """
        super(TheoreticalReceptiveField, self).__init__(dnn, layer, channel)
        
    def compute(self):
        if self.dnn.__class__.__name__ == 'AlexNet':
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

        theoretical_rf_size = 1
        #compute size based on net info
        for layer in reversed(range(self.net_struct['name'].index(self.mask.layers[0]) + 1)):
            kernel_size, stride, padding = self.net_struct['net'][layer]
            theoretical_rf_size = ((theoretical_rf_size - 1) * stride) + kernel_size
        return theoretical_rf_size
