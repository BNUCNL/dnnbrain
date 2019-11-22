#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:51:25 2019

@author: zhenzonglei
"""

#! /usr/bin/env python
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod

from skimage import segmentation 
from os.path import join as pjoin

from dnnbrain.dnn.base import ImageSet
from dnnbrain.dnn.core import Mask

class Algorithm(ABC):
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
        self.layer = layer
        self.channel = channel
        
class MinmalParcelImage(Algorithm):
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
   
    """
    def __init__(self, dnn, layer=None, channel=None):
       
       super(MinmalParcelImage, self).__init__(dnn, layer, channel)
       self.parcel = None
       
    
    def set_params(self, meth='SLIC', criterion='max'):
        """Set parameter for searching minmal image"""
        self.criterion = criterion


    def felzenszwalb_decompose(self, image, scale=100, sigma=0.5, min_size=50):
        """
        decompose images to several segments and put each parcel into a separated image
        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        meth[str]: method to decompose images
        
        Return:
        ------
        segments[ndarray]: Integer mask indicating segment labels.
        
        """   
        self.parcl = segmentation.felzenszwalb(image, scale, sigma, min_size)
        return self.parcel

        
    def slic_decompose(self, image, n_segments=250, compactness=10, sigma=1):
        """
        decompose images to several segments and put each parcel into a separated image
        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        meth[str]: method to decompose images
        
        Return:
        ------
        segments[ndarray]: Integer mask indicating segment labels.
        
        """   
        self.parcel =segmentation.slic(image, n_segments, compactness, sigma)
        return self.parcel

    
    def quickshift_decompose(self, image, kernel_size=3, max_dist=6, ratio=0.5):
        """
        decompose images to several segments and put each parcel into 
        a separated image with a black background
        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        meth[str]: method to decompose images
        
        Return:
        ------
        segments[ndarray]: Integer mask indicating segment labels.
        
        """   
        self.parcel = segmentation.quickshift(image, kernel_size, max_dist, ratio)
        
        return self.parcel
    
   
    
    def sort_parcel(self, order=''):
        """sort the parcel according the activation of dnn. order, ascending or descendign"""
        
        dnn_acts = self.dnn.compute_activation(patch_all, self.dmask).pool('max').get(self.layer)
        act_all = dnn_acts.flatten()
        
        #sort the activation from max to min
        act_sorted = np.argsort(-act_all)
        
        
        dnn_acts = self.dnn.compute_activation(patch_all, dmask)
        dnn_acts = dnn_acts.pool('max')
        dnn_acts = dnn_acts.get(layer)
    
        #put the dim from 4 to 1
        act_all = dnn_acts.flatten()
        
        #sort the activation from max to min
        act_sorted = np.argsort(-act_all)
        
        patch_add = self.generate_patch_mix(image, segments, act_sorted, RGB)
        
        dnn_acts_add = self.dnn.compute_activation(patch_add, dmask)
        dnn_acts_add = dnn_acts_add.pool('max')
        dnn_acts_add = dnn_acts_add.get(layer)
        
        act_add = dnn_acts_add.flatten()
        
        #find the max index
        act_max_index = np.argmax(act_add)

        #find the original index in act_sorted
        act_back_index = act_sorted[act_max_index] #!!!!
    
    
    def combine_parcel(self, index):
    """combine the indexed parcel into a image"""
        pass 
    
    def generate_minmal_image(self)):
        """
        Generate minimal image for a image. First sort the parcel by the activiton and 
        then iterate to find the best conbination of the parcel to get maximum activaiton
        
        Parameter:
        ---------
        stim[Stimulus]: stimulus
        Return:
    
        ------
        image_min_all[list]: all minimal images
        """    
        
        
        # workflow
        
        # sort the image
        # iterater combine image to get activation
        # return the opimized curve and minmal image
        pass
        
class MinmalComponentImage(Algorithm):
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """
    
    def set_params(self,  meth='pca', criterion='max'):
        """Set parameter for the estimator"""
        self.meth = meth
        self.criterion = criterion

    def compute(self, image):
        """Generate minmal image for image listed in stim object """
        pass