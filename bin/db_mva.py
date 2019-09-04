#! /usr/bin/env python

"""
Multivarate analysis to explore relations between CNN activation and response 
from brain/behavior.

CNL @ BNU
"""

import argparse
import numpy as np
#import pandas as pd
#import torch
from torch.utils.data import DataLoader, TensorDataset
#from scipy.stats import pearsonr
#from scipy.signal import convovle
from torchvision import transforms
from os.path import join as pjoin
from dnnbrain.dnn import analyzer
from dnnbrain.dnn import io as dio
from dnnbrain.brain import io as bio
from sklearn import linear_model, model_selection, decomposition, svm

    

def main():
    parser = argparse.ArgumentParser(description='Use CNN activation to \
                                     predict brain activation')
    
    parser.add_argument('-net',
                        type=str,
                        required=True,
                        metavar='NetName',
                        help='convolutional network name')
    
    parser.add_argument('-layer',
                        type=str,
                        required=True,
                        metavar='LayerName',
                        help='The layer whose activation is used \
                        to predict brain activity. For example, conv1 \
                        represents the first convolution layer, and  \
                        fc1 represents the first full connection layer.')
    
    parser.add_argument('-axis',
                        type=str,
                        default='layer',
                        required=True,
                        choices=['layer','channel', 'column'],
                        metavar='AxisName',
                        help='Target axis to organize the predictors. \
                        layer: do mva using all unit within a layer\
                        channel：do mva using all units within each channel. \
                        column: do mva using all units within each column. \
                        default is layer.')
    
    parser.add_argument('-dmask', 
                        type=str,
                        required=False,
                        metavar='DnnMaskFile',
                        help='a db.csv file in which channles of intereset \
                        and columns of interest are listed.')
    
    parser.add_argument('-dfe', 
                        type=str,
                        required=False,
                        metavar='DnnFeatureExtraction',
                        choices=['hist', 'max', 'mean','median'],
                        help='Feature extraction for dnn activation in \
                        the specified axis \
                        max: use max activiton as feature, \
                        median: use median activation as feature, \
                        mean: use mean activtion as feature, \
                        hist: use hist proflies as features.')  
    
    parser.add_argument('-dpca',
                        type=int,
                        required=False,
                        metavar='PCA',
                        help='The number of PC to be kept.') 
    
    parser.add_argument('-stim',
                        type=str,
                        required=False,
                        metavar='StimuliInfoFile',
                        help='a db.csv file provide stimuli information')
    
    parser.add_argument('-movie',
                        type=str,
                        required=False,
                        action='append',
                        metavar='MoiveStimulusFile',
                        help='a mp4 video file')
    
    parser.add_argument('-response',
                        type=str,
                        required=True,
                        metavar='ResponseFile',
                        help='a db.csv file to provide target response. \
			            The target reponse could be behavior measures or \
                         brain response from some rois')
    
    parser.add_argument('-hrf',
                        action='store_true',
                        required=False,
                        help='The canonical HRF is used.')
    
    parser.add_argument('-bmask',
                        type=str,
                        required=False,
                        metavar='BrainMaskFile',
                        help='Brain mask(nii or nii.gz) to indicate \
                        the voxel of interest.It works only when response \
                        is nii or nii.gz file')
          
    parser.add_argument('-model',
                        type=str,
                        required=True,
                        metavar='Model',
                        choices=['glm', 'lasso', 'svc','lrc'],
                        help='glm: general linear regression \
                        lasso: lasso regression \
                        svc: support vector machine classification \
                        lrc: logistic regression for classification.')
        
    parser.add_argument('-cvfold',
                        default=2,
                        type=int,
                        required=False,
                        metavar='FoldNumber',
                        help='cross validation fold number')
    
    parser.add_argument('-outdir',
                        type=str,
                        required=True,
                        metavar='OutputDir',
                        help='output directory. Model, accuracy, and related.')
    
    args = parser.parse_args()
    
    #%% Brain/behavior response(i.e.,Y)
    """
    First, we prepare the response data for exploring relations betwwen 
    the CNN activation and brain/behavior responses.  
    
    """
    if args.response.endswith('db.csv'):
          resp = dio.read_dbcsv(args.response)
          resp = resp['VariableName'].values() # n_stim x n_roi
            
    elif args.response.endswith('nii') or args.response.endswith('nii.gz'):
        resp, header = bio.load_brainimg(args.response)
    else:
        raise Exception('Only db.csv and nii vloume are supported')
        
    # Get tr from nii header                
    tr = header['pixdim'][4]
    if header.get_xyzt_units()[-1] == 'ms':
                tr = tr/ 1000
        
    # Get resp data within brain mask
    resp = resp.reshape(resp.shape[0],-1)  # n_stim x n_vox
    if args.bmask is None:    
        bmask = np.any(resp,0)
    else:
        bmask, _ = bio.load_brainimg(args.bmask, ismask=True)            
        bmask = bmask.reshape(-1)    
        assert bmask.shape[0] == resp.shape[1], "mask and response mismatched in space"
        
    Y = resp[:, bmask] # n_stim x n_roi or n_vox
    
    #%% CNN activation
    """
    Second, we prepare CNN activation(i.e., X) for exploring relations betwwen 
    the CNN activation and brain/behavior responses. 
    
    """
    # Load CNN
    netloader = dio.NetLoader(args.net)
    transform = transforms.Compose([transforms.Resize(netloader.img_size),
                                    transforms.ToTensor()])  
    # Load stimulus
    stim = dio.read_dbcsv(args.stim)
    stim_path, stim_id = stim['picpath'], stim['VariableName']            
    picdataset = dio.PicDataset(stim_path,stim_id,transform=transform)
    picdataloader = DataLoader(picdataset, batch_size=8, shuffle=False)
    
    # calculate dnn activation: n_stim * n_channel * unit * unit
    dnn_act = analyzer.dnn_activation(picdataloader, args.net, args.layer)
    # n_stim * n_channel * n_unit
    dnn_act = dnn_act.reshape(dnn_act.shape[0], dnn_act.shape[1], -1)
    
     # define dnn mask
    if args.dmask is not None:
        dmask = dio.read_dbcsv(args.dmask)
        chnoi = dmask['VariableName']['chn']
        coloi = dmask['VariableName']['col']
        dnn_act = dnn_act[:,chnoi,coloi] # n_stim x n_chnoi x n_coloi

    # dnn feature extraction
    if args.dfe is not None:    
        if args.axis == 'channel':
            if args.dfe == 'mean':
                dnn_act = np.mean(dnn_act,axis=-1)[...,None]
            elif args.dfe == 'max':
                dnn_act = np.max(dnn_act,axis=-1)[...,None]
            elif args.dfe == 'median':
                dnn_act = np.median(dnn_act,axis=-1)[...,None]
            elif args.dfe == 'hist':
                dnn_act = np.histogramdd(dnn_act)

                
        elif args.axis == 'column':
            if args.dfe == 'mean':
                dnn_act = np.mean(dnn_act,axis=1)[...,None]
            elif args.dfe == 'max':
                dnn_act = np.max(dnn_act,axis=1)[...,None]
            elif args.dfe == 'median':
                dnn_act = np.median(dnn_act,axis=1)[...,None]
            elif args.dfe == 'hist':
                dnn_act = np.histogramdd(dnn_act)
                
    # size of cnn activation            
    n_stim,n_chn,n_unit = dnn_act.shape
  
  # PCA on dnn features
    if args.pca is not None:             
        pca = decomposition.PCA(n_components=args.pca)
        if args.axis == 'layer':       
            X = dnn_act.reshape(n_stim,-1) 
            X = pca.fit_transform(X)

        elif args.axis == 'channel':
            X = np.zeros((n_stim, n_chn, args.pca))
            for chn in range(n_chn):
                X[:,chn,:] = pca.fit_transform(X[:,chn,:])
                
        elif args.axis == 'column':
            X = np.zeros((n_stim, args.pca, n_unit))
            for unit in range(n_unit):
                X[:,:,unit] = pca.fit_transform(X[:,:,unit])
            
            
     # Convert dnn activtaion to bold response.
    if args.hrf is not None:
         onset,duration = stim_id['onset'], stim_id['duration']
         X = analyzer.generate_bold_regressor(X,onset,
                                              duration,resp.shape[0],tr)
         
   
    #%% multivariate analysis
    """
    Third, we use multivariate model to explore the relations between 
    CNN activation and brain/behavior responses. 
    
    """
    
    if args.model == 'glm':
        model = linear_model.LinearRegression()
    elif args.model == 'lasso':
        model = linear_model.Lasso()
    elif args.model == 'svc':
        model = svm.SVC(kernel="linear", C=0.025)
    elif args.model == 'lrc': 
        model = linear_model.LogisticRegression()
    else:
        raise Exception('Please use glm and lasso for linear regression and \
                            use svc and lrc for linear classification')
    
  
    # run mv model to do prediction
    model.fit(X,Y)
    pred = model.predict(X)
    
    
    # validate the mv model
    if args.model == 'glm' or args.model == 'lasso':        
         scoring='explained_variance'
    else:
         scoring='accuracy'
    score = [model_selection.cross_val_score(
            model,X,Y[:,i],scoring,cv=args.cvfold) for i in range(Y.shape[1])]
    
    
     #%% save the results to disk
    """
    Finally, we save the related results to the outdir.
     
    """
    if args.response.endswith('db.csv'):
          resp = dio.save_dbcsv(fname)
            
    else:
        
        bio.save_brainimg(pjoin(args.outdir, 'voxel_score.nii.gz'), out_brainimg, header)
        
if __name__ == '__main__':
    main()
