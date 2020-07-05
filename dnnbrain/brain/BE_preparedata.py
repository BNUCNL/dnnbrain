#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:00:25 2020

@author: gongzhengxin  zhouming
"""
import numpy as np
from dnnbrain.dnn.core import Mask

def gen_feature(dnn, stimuli, layer, chn='all', mask=None):
    """
    Generate stimuli features
    
    Parameters
    ----------
    dnn[DNN]
    stimuli[ndarray]: shape(n_stim, n_chn, height, width)
    layer[list]: its elements are different layer names
    chn[str|list]: channels that all the layers share
        if chn is str, it must be 'all' which means all channels
        if chn is list, its elements are serial numbers of channels
        default is 'all'
    mask[dict]: storing the layer and chn info 
        layer is the key name and chn is the value
        Their requirements are the same as above
        Note: if using mask, please do not using 'layer' and 'chn' parameters
        
    Return:
    ---------
    features[list]: length equal to layer num
        each element is a activation ndarray for corresponding layer
    """
    # initialize some params
    features = []
    dmask = Mask()
    # start computing
    if mask is not None:   
        if any([layer, chn]):
            raise AssertionError('Do not define layer and chn if you use mask!')
        else:
            for layer_name in mask.keys():
                dmask.set(layer_name, channels=mask[layer_name])
                act = dnn.compute_activation(stimuli, dmask).get(layer_name)
                # handle problems of too many feature maps
                if act.shape[1] > 1024:
                    varr = np.var(act, axis=0).squeeze()
                    index = np.argsort(-varr)[:1024]
                    act = act[:,index,:,:]
                features.append(act)
                dmask.clear()
    else:
        for layer_name in layer:
            dmask.set(layer_name, channels=chn)
            act = dnn.compute_activation(stimuli, dmask).get(layer_name)
            # handle problems of too many feature maps
            if act.shape[1] > 1024:
                varr = np.var(act, axis=0).squeeze()
                index = np.argsort(-varr)[:1024]
                act = act[:,index,:,:]
            features.append(act)
            dmask.clear()
    return features

def gen_rf(layers):
    
    pass

def gen_dataset(fmaps,rfmasks):
    
    pass
