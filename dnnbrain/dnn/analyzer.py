
import torch
import torchvision
from torch import nn
from torchvision import models
from torchvision import transforms, utils
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


import os
import pandas as pd    
import numpy as np
from dnnbrain.dnn.models import dnn_truncate, TransferredNet, dnn_train_model
from dnnbrain.dnn import io as iofiles
from scipy.stats import pearsonr
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
from dnnbrain.dnn import io as dnn_io
from dnnbrain.brain import io as brain_io
from scipy.signal import convovle
from sklearn import linear_model, model_selection, decomposition, svm

    


def dnn_activation(input, netname, layer, channel=None):
    """
    Extract DNN activation

    Parameters:
    ------------
    input[dataloader]: input image dataloader	
    netname[str]: DNN network
    layer[str]: layer name of a DNN network
    channel[list]: specify channel in layer of DNN network, channel was counted from 1 (not 0)

    Returns:
    ---------
    dnnact[numpy.array]: DNN activation, A 4D dataset with its format as pic*channel*unit*unit
    """
    loader = iofiles.NetLoader(netname)
    actmodel = dnn_truncate(loader, layer)
    actmodel.eval()
    dnnact = []
    for picdata, target in input:
        dnnact_part = actmodel(picdata)
        dnnact.extend(dnnact_part.detach().numpy())
    dnnact = np.array(dnnact)

    if channel:
        channel_new = [cl - 1 for cl in channel]
        dnnact = dnnact[:, channel_new, :, :]
    return dnnact
	
    #%% multivarient prediction analysis 
    # func
    def dnn_bold_regressor(activation,timing,n_vol,tr):
        '''convolve dnn_act with hrf and align with timeline of response
        
        parameters:
        ----------
            cond: Nx2 np array, the first column is timing, second column is 
            cond id
            n_vol: int, number of volume
            tr: double, sampling time in sec for a volume
            
            x: [n_event,n_sample]
                Onset, duration and x' 1st dim should have the same size.
            resp: total 1-d array
            tr: in sec
            
        '''
        hrf = spm_hrf(tr,25)
        reg =  convolve(X, hrf)
        reg_names = 'cond_name'
        