try:
    import torch
    import torchvision
    from torch import nn
    from torchvision import models
    from torchvision import transforms, utils
	from torch.utils.data import DataLoader, TensorDataset
	from torchvision import transforms
except ModuleNotFoundError:
    raise Exception('Please install pytorch and torchvision in your work station')

import os
import pandas as pd    
import numpy as np
from dnnbrain.dnn.models import dnn_truncate, TransferredNet, dnn_train_model
from dnnbrain.dnn import io as iofiles
from scipy.stats import pearsonr
from nipy.modalities.fmri.hemodynamic_models import compute_regressor
from dnnbrain.dnn import io as dnn_io
from dnnbrain.brain import io as brain_io

try:
    from sklearn import linear_model, model_selection, decomposition, svm
except ModuleNotFoundError:
    raise Exception('Please install sklearn in your workstation')



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
	
	
	
	
	
def db_mv_pred(axis=None,dnn_roi_filelist=None,mask=None,
               unit_reduction=None,pca=None,cvfold=2,pred_model,hrf=None,
               stim,net,layer,response,outdir,):    

    """
    perform multivarient regression or classification

    Parameters:
    ------------
	sitm: a csv file contains picture stimuli path and picture onset time.
		[PicPath]
		stimID,		onset
		face1.png,	1.1
		face2.png,	3.1
		scene1.png,	5.1
	response: variale(s) to be predicted.
		format: csv files or nifti files.
		csv file format:
			[data type: time course or ramdom samples]
			[tr: in sec]
			roiname,		roiname
			51741.1,		57739.1
			51823.3,		57831.5
			52353.2,		57257.3		
	mask: brain mask for response if response is brain imaging data.   
    netname: DNN network
    layer: layer name of a DNN network
    axis: specify an axis to organize the predictors..
		channel：A model will be built for each channel using all the units in that channel. 
		column: A model will be built for each column using all the channels in that column.
		default is None.
	dnn_roi_filelist: a csv file contains channels or units of interest to build the model.
		channel,	unit
		1,			2
		3,			3
		15
	unit reduction: method to summary units into channel level before building the model
		mean or max. Default is None.
	pred_model: specify a method to build the model
		glm_r: general linear regression
		svm_r: support vector machine regression
		lasso_r: lasso regression
		svm_c: support vector machine classification
	pca: the number of principle components to perform a PCA on the predictors before
		building the model.
		Default is 10
	cvfold: cross-validation fold
		Default is 2
	hrf: specify a hrf model to perform convolution on predictors.
		spm: this is the hrf model used in spm
		spm_time: this is the spm model plus its time derivative (2 regressors)
		spm_time_dispersion: idem, plus dispersion derivative (3 regressors)
		canonical: this one corresponds to the Glover hrf
		canonical_derivative: the Glover hrf + time derivative (2 regressors)
		fir: finite impulse response basis, a set of delayed dirac models
	outdir: output directory
    """

    #%% Load response(y)
    def load_resp_csv(csv_path):        
        with open(csv_path,'r') as f:
            meta_data = [x.rstrip() for i, x in enumerate(f) if i<=2]
            resp_type = meta_data[0]
            tr = np.float(meta_data[1])
        resp_data = pd.read_csv(csv_path, skiprows=2)
        return resp_type, tr, list(resp_data.keys()), resp_data.values
    
    if response.endswith('csv'):
        assert mask is None, "Loading .csv response does not need a mask."
        # npic * ROIs
        resp_type,tr,roi_keys,resp = load_resp_csv(response)       
    elif response.endswith('nii') or response.endswith('nii.gz'):
        # Load brain images
        resp_raw, header = brain_io.load_brainimg(response)
        
        # get tr from nifti header
        if hrf is not None:
            assert header.get_xyzt_units()[-1] is not None, "TR was not provided in the brain imaging file header"
            if header.get_xyzt_units()[-1] in ['s','sec']:
                tr = header['pixdim'][4]
            elif header.get_xyzt_units()[-1] == 'ms':
                tr = header['pixdim'][4] / 1000
        
        # get masked resp data
        resp_raw_shape = np.shape(resp_raw)
        resp_raw = resp_raw.reshape(resp_raw_shape[0],-1)       
        if mask is not None:
            brain_mask, _ = brain_io.load_brainimg(mask, ismask=True)
            assert np.shape(brain_mask) == resp_raw_shape[1:], "Mask and brainimg should have the same geometry shape"
            brain_mask = brain_mask.reshape(-1)    
        else:
            brain_mask = np.zeros(resp_raw.shape[1])
            brain_mask[resp_raw.mean(0)!=0] = 1        
        resp = resp_raw[:,brain_mask!=0]
        
    else:
        raise Exception('Not support yet, please contact to the author for implementation.')

    brain_roi_size = resp.shape[1]    
    print('response data loaded')
    
    
    #%% Get CNN activation(x)
    netloader = dnn_io.NetLoader(net)
    imgcropsize = netloader.img_size     
    transform = transforms.Compose([transforms.Resize(imgcropsize),
                                    transforms.ToTensor()])                            
    picdataset = dnn_io.PicDataset(stim, transform=transform)
    assert 'stimID' in picdataset.csv_file.keys(), 'stimID must be provided in stimuli csv file'
    assert 'onset' in picdataset.csv_file.keys(), 'onset must be provided in stimuli csv file'
    assert 'duration' in picdataset.csv_file.keys(), 'duration must be provided in stimuli csv file'
 
    picdataloader = DataLoader(picdataset, batch_size=8, shuffle=False)
        # dnn_act: pic * channel * unit * unit
    
    # read dnn roi    
    chn_roi = None
    unit_roi = None
    if dnn_roi_filelist is not None:
        dnn_roi = pd.read_csv(dnn_roi_filelist)
        if 'channel' in dnn_roi.keys():
            chn_roi = dnn_roi['channel'].values
            chn_roi = np.asarray(chn_roi[~np.isnan(chn_roi)] - 1, dtype=np.int)
        if 'unit' in dnn_roi.keys():
            unit_roi = dnn_roi['unit'].values
            unit_roi = np.asarray(unit_roi[~np.isnan(unit_roi)] - 1, 
                                           dtype=np.int)
    # get dnn activation of dnn roi        
    dnn_act = dnn_activation(
            picdataloader, net, layer, channel=list(chn_roi))
    dnn_act = dnn_act.reshape(dnn_act.shape[0], dnn_act.shape[1], -1)
    
    if unit_roi is not None:
        dnn_act = dnn_act[:,:,unit_roi]
           
    # unit dimention reduction
    if unit_reduc is not None:
        if unit_reduc == 'mean':
            dnn_act = dnn_act.mean(-1)[:,:,np.newaxis]
        elif unit_reduc == 'max':
            dnn_act = dnn_act.max(-1)[:,:,np.newaxis]
    
    n_stim = dnn_act.shape[0]
    n_chn = dnn_act.shape[1]
    n_unit = dnn_act.shape[2]
    
    print('dnn activation generated')
    

    #%% multivarient prediction analysis 
    # func
    def x_hrf(stim_csv_pd,x,hrf_model,fmri_frames,tr):
        '''convolve dnn_act with hrf and align with timeline of response
        
        parameters:
        ----------
            stim_csv_pd: pandas dataframe, with onset and duration keys.
            x: [n_event,n_sample]
                Onset, duration and x' 1st dim should have the same size.
            resp: total 1-d array
            tr: in sec
            
        '''
        x_hrfed = []
        for i in range(x.shape[1]):            
            exp_condition = [
                    stim_csv_pd['onset'],stim_csv_pd['duration'],x[:,i]]
            frametimes = np.arange(fmri_frames) * tr
            regressor,_ = compute_regressor(exp_condition,hrf_model,frametimes)            
            x_hrfed.append(regressor)
        
        x_hrfed = np.squeeze(np.asarray(x_hrfed)).transpose(1,0)
        return x_hrfed
    

    def dim2(x,axis=1):
        if np.ndim(x) == 1:
            return np.expand_dims(x,axis=axis)
        else:
            return x
        
    def x_pca(x,n_components):
        pca_m = decomposition.PCA(n_components)
        pc = pca_m.fit_transform(x)
        return pc,pca_m
        
    
    # prediction models
    def glm_r(x,y,cvfold):
        '''linear model using ordinary least squares
        
        parameters:
        -----------
        x: [n_samples, n_features]
        y: [n_samples, n_resp_variable]
        
        '''  
        
        # model score
        model = linear_model.LinearRegression()
        m_score = [model_selection.cross_val_score(
                model,x,y[:,y_i],scoring='explained_variance',cv=cvfold
                ) for y_i in range(y.shape[1])]
        m_score = dim2(np.asarray(m_score).mean(-1),axis=0)
        
        # output data
        model = linear_model.LinearRegression()
        model.fit(x, y)
        m_pred = model.predict(x)       
       
        return model, m_score, m_pred


    def lasso_r(x,y,cvfold):
        pass
    
    def svm_r(x,y,cvfold):
        pass
    
    def svm_c(x,y,cvfold):
        pass
    
    def lda_c(x,y,cvfold):
        pass
            

 
    def mv_model(x,y,pred_model,cvfold):

        if pred_model == 'glm':
            model, m_score, m_pred = glm_r(x,y,cvfold)
            return model, m_score, m_pred
        
        elif pred_model == 'lasso':
            lasso_r(x,y,cvfold)
            
        elif pred_model == 'svr':
            svm_r(x,y,cvfold)
                
        elif pred_model == 'svc':
            if response.endswith('nii') or response.endswith('nii.gz'):
                raise Exception('Classification is not supported with input as brain images.')
            svm_c(x,y,cvfold)
#        elif model == 'lda':
#            if response.endswith('nii') or response.endswith('nii.gz'):
#                raise Exception('Classification is not supported with input as brain images.')
#            model = lda.()
#            score_evl = 'accuracy'
        else:
            raise Exception('Please select lmr or lmc for univariate prediction analysis.')
            
               
    # mv main analysis
    if axis is None:                
        # pca on x
        if pca is None:
            x = dnn_act.reshape(n_stim,-1) # dnn_act: [n_stim, n_chn * n_unit]
        else: 
            x, pca_m = x_pca(dnn_act.reshape(dnn_act.shape[0],-1),pca)
        
        # hrf convolve, should be performed after pca
        if hrf is not None:
             x = x_hrf(picdataset.csv_file,x,hrf,resp.shape[0],tr)
           
        # prediction model
        mv_model(x=dnn_act,y=dim2(resp,1),pred_model=pred_model,cvfold=cvfold)
        
                   
    elif axis == 'channel':       
        for chn in range(dnn_act.shape[1]):
            # pca on x
            if pca is None:
                x = dnn_act.reshape(n_stim,-1) # dnn_act: [n_stim, n_chn * n_unit]
            else: 
                x, pca_m = x_pca(dnn_act.reshape(dnn_act.shape[0],-1),pca)
            
            # hrf convolve, should be performed after pca
            if hrf is not None:
                 x = x_hrf(picdataset.csv_file,x,hrf,resp.shape[0],tr)

        
    elif axis == 'column':
        for unit in range(dnn_act.shape[2]):
            # pca on x
            if pca is None:
                x = dnn_act.reshape(n_stim,-1) # dnn_act: [n_stim, n_chn * n_unit]
            else: 
                x, pca_m = x_pca(dnn_act.reshape(dnn_act.shape[0],-1),pca)
            
            # hrf convolve, should be performed after pca
            if hrf is not None:
                 x = x_hrf(picdataset.csv_file,x,hrf,resp.shape[0],tr)
        