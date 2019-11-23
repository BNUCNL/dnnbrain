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
from dnnbrain.dnn.core import Mask, Algorithm


        
class MinmalParcelImage(Algorithm):
    """
    A class to generate minmal image for target channels from a DNN model 
   
    """
    def __init__(self, dnn, layer=None, channel=None):
       
       super(MinmalParcelImage, self).__init__(dnn, layer, channel)
       self.parcel = None
       
    
    def set_params(self, activiton_criterion, search_criterion='max'):
        """
        Set parameter for searching minmal image
        criterion: criterion to 
        """
        self.activation_criterion = activation_criterion
        self.search_criterion = search_criterion


    def felzenszwalb_decompose(self, image, scale=100, sigma=0.5, min_size=50):
        """
        Decompose image to multiple parcels using felzenszwalb method and
        put each parcel into a separated image with a black background
        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        
        Return:
        ------
        segments[ndarray]: Integer mask indicating segment labels.
        
        """   
        self.parcel = segmentation.felzenszwalb(image, scale, sigma, min_size)
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
        ------
        segments[ndarray]: Integer mask indicating segment labels.
        
        """   
        self.parcel =segmentation.slic(image, n_segments, compactness, sigma)
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
        ------
        segments[ndarray]: Integer mask indicating segment labels.
        
        """   
        self.parcel = segmentation.quickshift(image, kernel_size, max_dist, ratio)
        return self.parcel
    
    def sort_parcel(self, order='descending'):
        """
        sort the parcel according the activation of dnn. 
        order[str]: ascending or descending
        """
        
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
    
    def generate_minmal_image(self):
        """
        Generate minimal image. We first sort the parcel by the activiton and 
        then iterate to find the combination of the parcels which can maximally
        activate the target channel.
        
        Note: before call this method, you should call xx_decompose method to 
        decompose the image into parcels. 
        
        Parameter:
        ---------
        stim[Stimulus]: stimulus
        Return:
    
        ------
        image_min_all[list]: all minimal images
        """
        
        if self.parcel is None: 
            raise AssertionError('Please run decompose method to '
                                 'decompose the image into parcels')
                    
        
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
        
        
    def pca_decompose(self):
        pass
    
    
    def ica_decompose(self):
        pass
    
    
    
    def sort_componet(self, order='descending'):
        """
        sort the component according the activation of dnn. 
        order[str]: ascending or descending
        """
      
    def combine_component(self, index):
        """combine the indexed component into a image"""
        pass 
    
    
    
    def generate_minmal_image(self):
        """
        Generate minimal image. We first sort the component by the activiton and 
        then iterate to find the combination of the components which can maximally
        activate the target channel.
        
        Note: before call this method, you should call xx_decompose method to 
        decompose the image into parcels. 
        
        Parameter:
        ---------
        stim[Stimulus]: stimulus
        Return:
    
        ------