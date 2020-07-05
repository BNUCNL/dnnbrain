#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:00:25 2020

@author: gongzhengxin  zhouming
"""
import numpy as np
def gen_feature(dnn,stimuli,layers,channels):
    
    pass

def gen_rf(features,x,y,sigma):
    """
    parameters:
    -----------
    features: list, store of featrue maps from DNN or else
    x,y: float, location of rf center
    sigma: float, rf size
    
    return:
    -----------
    rfs: list, rf masks of each map in features  
    """
    rfs = []
    for i, f in enumerate(features):
        
        if len(f.shape)==2:
            f = f.reshape(f.shape+(1,1))
            
        n_sample = f.shape[0]
        n_dimen = f.shape[1]
        fmap_size = f.shape[2]
        
        rf = gauss(x,y,fmap_size,sigma)
        rf = np.tile(rf,(n_dimen*n_sample,1)).reshape(n_sample,n_dimen,fmap_size,fmap_size)
        
        rfs += [rf,]
    
    return rfs


def gen_dataset(fmaps,rfmasks):
    
    pass

def gauss(xi,yi,fmap_size, sigma):
    """
    parameters
    ----------
    xi,yi: float, center location of rf
    fmap_size: int, size of feature map
    sigma: float, gaussian radius 
    
    return:
    ----------
    mask: ndarray, gaussian mask 
    """
    mask = np.zeros((fmap_size, fmap_size))
    center = fmap_size//2
    if sigma<=0:
        sigma = ((fmap_size-1)*0.5-1)*0.3+0.8 # opCV default
        
    s = sigma**2
    coef = 1/(2*np.pi*s)
    sum_val =  0
    for i in range(fmap_size):
        for j in range(fmap_size):
            x, y = center - i, j - center
            
            mask[i, j] = coef*np.exp(-((x-xi)**2+(y-yi)**2)/2*s)
            sum_val += mask[i, j]
    
    mask = mask/sum_val
    
    return mask
























