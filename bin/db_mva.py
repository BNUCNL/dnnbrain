#! /usr/bin/env python

"""
Multivarate analysis to explore relations between CNN activation and response 
from brain/behavior.

CNL @ BNU
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from os.path import join as pjoin
from scipy.stats import pearsonr
from scipy.signal import convovle
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
                        required=True,
                        choices=['layer','channel', 'column'],
                        metavar='AxisName',
                        help='specify an axis to organize the predictors \
                        channel：A model will be built for each channel using \
                        all the units in that channel. \
                        column: A model will be built for each column using \
                        all the channels in that column. \
                        default is None.')
    
    parser.add_argument('-dmask', 
                        type=str,
                        required=False,
                        metavar='DnnMaskFile',
                        help='a csv file contains channels or \
                        units of interest to build the model in two rows. \
                        the first row is channel of interest, the second rows \
                        is column of intereset.')
    
    parser.add_argument('-dfe', 
                        type=str,
                        required=False,
                        metavar='DnnFeatureExtraction',
                        choices=['hist', 'max', 'mean','median'],
                        help='Do feature extraction from raw dnn activation in \
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
                        help='a csv file contains picture stimuli path and \
                        picture onset time. \
                        [PicPath] \
                        stimID,		onset \
		                face1.png,	1.1 \
		                face2.png,	3.1 \
		                scene1.png,	5.1')
    
    parser.add_argument('-movie',
                        type=str,
                        required=False,
                        action='append',
                        metavar='MoiveStimulusFile',
                        help='a mp4 video file')
    
    parser.add_argument('-response',
                        type=str,
                        required=True,
                        metavar='BrainActFile',
                        help='response: variale(s) to be predicted.\
                        format: csv files or nifti files.\
		                csv file format: \
			            [data type: time course or ramdom samples] \
			            [tr: in sec] \
			            roiname,		roiname \
			            51741.1,		57739.1 \
			            51823.3,		57831.5 \
			            52353.2,		57257.3')
    
    parser.add_argument('-hrf',
                        action='store_true',
                        required=False,
                        help='The canonical HRF is used.')
    
    parser.add_argument('-bmask',
                        type=str,
                        required=False,
                        metavar='BrainMaskFile',
                        help='brain mask used to extract activation locally.')
          
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
    if response.endswith('csv'):
          resp_type,tr,roi_keys,resp = bio.load_resp_csv(response)    
          if bmask:
              print('Brain mask is ignored')
            
    elif response.endswith('nii') or response.endswith('nii.gz'):
        # Load nii images
        resp_raw, header = bio.load_brainimg(response)
        
        # Get tr from nii header
        if args.hrf is not None:
            assert header.get_xyzt_units()[-1] is not None,...
            "TR was not provided in the brain imaging file header"           
            if header.get_xyzt_units()[-1] in ['s','sec']:
                tr = header['pixdim'][4]
            elif header.get_xyzt_units()[-1] == 'ms':
                tr = header['pixdim'][4] / 1000
        
        # Get resp data within bmask
        resp_raw_shape = np.shape(resp_raw)
        resp_raw = resp_raw.reshape(resp_raw_shape[0],-1)       
        if args.bmask is not None:
            brain_mask, _ = bio.load_brainimg(mask, ismask=True)
            assert np.shape(brain_mask) == resp_raw_shape[1:], ...
            "Mask and brainimg should have the same geometry shape"
            brain_mask = brain_mask.reshape(-1)    
        else:
            brain_mask = np.zeros(resp_raw.shape[1])
            brain_mask[resp_raw.mean(0)!=0] = 1        
        resp = resp_raw[:,brain_mask!=0]
        
    else:
        raise Exception('Please provide csv file for roi analyais, \
                        nii or nii.gz for brain voxel-wise mapping')
    brain_roi_size = resp.shape[1]    
    
    
    
    #%% CNN activation
    """
    Second, we prepare CNN activation(i.e., X) for exploring relations betwwen 
    the CNN activation and brain/behavior responses. 
    
    """
    # Load CNN
    netloader = dio.NetLoader(args.net)
    imgcropsize = netloader.img_size     
    transform = transforms.Compose([transforms.Resize(imgcropsize),
                                    transforms.ToTensor()])  
    # Load stimulus                      
    picdataset = dio.PicDataset(args.stim, transform=transform)
    assert 'stimID' in picdataset.csv_file.keys(), ...
    'stimID must be provided in stimuli csv file'
    assert 'onset' in picdataset.csv_file.keys(), ...
    'onset must be provided in stimuli csv file'
    assert 'duration' in picdataset.csv_file.keys(), ...
    'duration must be provided in stimuli csv file'
    picdataloader = DataLoader(picdataset, batch_size=8, shuffle=False)
    
   
    # calculate dnn activation: pic * channel * unit * unit
    dnn_act = analyzer.dnn_activation(picdataloader, args.net, args.layer)
    dnn_act = dnn_act.reshape(dnn_act.shape[0], dnn_act.shape[1], -1)
    
     # define dnn mask
    if args.dmask is not None:
        dmask = dio.read_dmask_csv(args.dmask)

    # data within dmask
    if args.axis == 'channel':
        dnn_act = dnn_act[:,:,dmask]
    elif args.axis == 'column':
        dnn_act = dnn_act[:,dmask,:]
    
    # dnn feature extraction
    if args.dfe is not None:    
        if args.axis == 'channel':
            if args.dfe == 'mean':
                dnn_act = dnn_act.mean(-1)[:,:,np.newaxis]
            elif args.dfe == 'max':
                dnn_act = dnn_act.max(-1)[:,:,np.newaxis]
            elif args.dfe == 'median':
                dnn_act = dnn_act.max(-1)[:,:,np.newaxis]
            elif args.dfe == 'hist':
                dnn_act = np.histogramdd(dnn_act)

                
        elif args.axis == 'channel':
            if args.dfe == 'mean':
                dnn_act = dnn_act.mean(1)
            elif args.dfe == 'max':
                dnn_act = dnn_act.max(1)
            elif args.dfe == 'median':
                dnn_act = dnn_act.max(1)
            elif args.dfe == 'hist':
                dnn_act = np.histogramdd(dnn_act,15)
                
                
    # size of cnn activation            
    n_stim,n_chn,n_unit = dnn_act.shape
  

   
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
        raise Exception('Please use glm and lasso for regression and \
                            use svc and lrc for classification')
    
    # PCA on dnn features
    if args.pca is not None:             
        pca = decomposition.PCA(n_components=args.pca)
        if args.axis == 'layer':       
            X = dnn_act.reshape(n_stim,-1) 
            X = pca.fit_transform(X)

        elif args.axis == 'channel':
            X = np.zeros((n_stim, n_chn, args.pca))
            for chn in range(n_chn):
                X(:,chn,:) = pca.fit_transform(X(:,chn,:)
                
        elif args.axis == 'column':
            X = np.zeros((n_stim, args.pca, n_unit))
            for unit in range(n_unit):
                X(:,:,unit) = pca.fit_transform(X(:,:,unit))
            
            
     # Convert dnn activtaion to bold response
     if args.hrf is not None:
         X = analyzer.dnn_bold_regressor(X, timing,tr)
         
     # run mv model
     m_score = [model_selection.cross_val_score(
                model,X,Y[:,Y_i],scoring='explained_variance',cv=cvfold
                ) for Y_i in range(Y.shape[1])]
     m_score = dim2(np.asarray(m_score).mean(-1),axis=0)
        
    # output data
    model.fit(X,Y)
    m_pred = model.predict(X)
    
    
     #%% save the results to disk
     """
     Finally, we save the related results to the outdir.
     
     """
     
     if args.roi:
        score_df = pd.DataFrame({'ROI': roilabel, 'scores': scores})
        # Save behavior measurement into hardware
        score_df.to_csv(pjoin(args.outdir, 'roi_score.csv'), index=False)
    else:
        # output image: channel*voxels
        out_brainimg = np.zeros((1, *brainimg_data.shape[1:]))
        for i, b_idx in enumerate(brainact_idx):
            out_brainimg[0, b_idx[0], b_idx[1], b_idx[2]] = scores[i]
        # Save image into hardware
        bio.save_brainimg(pjoin(args.outdir, 'voxel_score.nii.gz'), out_brainimg, header)


if __name__ == '__main__':
    main()
