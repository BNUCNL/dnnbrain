import numpy as np
from dnnbrain.dnn import io as iofiles
from dnnbrain.dnn.models import dnn_truncate
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
from scipy.signal import convolve


def dnn_activation(input, netname, layer, channel=None, column=None, 
                   fe_axis=None, fe_meth=None):
    """
    Extract DNN activation

    Parameters:
    ------------
    input[dataloader]: input image dataloader	
    netname[str]: DNN network
    layer[str]: layer name of a DNN network
    channel[list]: specify channel in layer of DNN network, channel was counted from 1 (not 0)
    column[list]: column of interest
    fe_axis{str}: axis for feature extraction
    fe_meth[str]: feature extraction method, max, mean, median

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
    dnnact = dnnact.reshape(dnnact.shape[0], dnnact.shape[1], -1)
    
    
    # mask the data
    if channel is not None:
        dnnact = dnnact[:,channel,:]
    
    if column is not None: 
        dnnact = dnnact[:, :, column]
        
    # feature extraction
    if (fe_axis is None) != (fe_meth is None):
        raise Exception('Please specify fe_axis and fe_meth at the same time.')
    
    if fe_axis == 'layer':
        a = None
    elif fe_axis == 'channel':
        a = 1
    elif fe_axis == 'column':
        a = -1
    else:
        raise Exception('fe_axis should be layer, channel or column')
        
    if fe_axis is not None:
        if fe_meth == 'max':
            dnnact = np.max(dnnact,a)
        elif fe_meth == 'mean':
            dnnact = np.mean(dnnact,a)
        elif fe_meth == 'median':
            dnnact = np.median(dnnact,a)
    
    return dnnact





def generate_bold_regressor(X, onset, duration, vol_num, tr, ops=100):
    """
    convolve event-format X with hrf and align with timeline of BOLD signal

    parameters:
    ----------
    X[array]: [n_event] or [n_event,n_sample]
    onset[list or array]: in sec. size = n_event
    duration[list or array]: list or array. in sec. size = n_event
    vol_num[int]: total volume number of BOLD signal
    tr[float]: repeat time in second
    ops[int]: oversampling number per second

    Returns:
    ---------
    X_hrfed[array]: same shape with X
    """
    assert ops in (10, 100, 1000), 'Oversampling rate must be one of the (10, 100, 1000)!'
    decimals = int(np.log10(ops))
    onset = np.round(np.asarray(onset), decimals=decimals)
    duration = np.round(np.asarray(duration), decimals=decimals)
    tr = np.round(tr, decimals=decimals)

    if np.ndim(X) == 1:
        X = X[:, np.newaxis]

    batch_size = int(100000 / ops)
    batches = np.arange(0, X.shape[-1], batch_size)
    batches = np.r_[batches, X.shape[-1]]

    # compute volume acqusition timing
    vol_t = (np.arange(vol_num) * tr * ops).astype(int)

    time_point_num = int((onset + duration).max() * ops)
    X_hrfed = np.zeros([vol_num, 0])
    for k, batch in enumerate(batches[:-1]):
        X_i = X[:, batch:batches[k+1]]
        # generate X raw time course
        X_tc = np.zeros((time_point_num, X_i.shape[-1]), dtype=np.float32)
        for i, onset_i in enumerate(onset):
            onset_i_start = int(onset_i * ops)
            onset_i_end = int(onset_i_start + duration[i] * ops)
            X_tc[onset_i_start:onset_i_end, :] = X_i[i, :]

        # generate hrf kernel
        hrf = spm_hrf(tr, oversampling=tr*ops)
        hrf = hrf[:, np.newaxis]

        # convolve X raw time course with hrf kernal
        X_tc_hrfed = convolve(X_tc, hrf, method='fft')

        # downsample to volume timing
        X_hrfed = np.c_[X_hrfed, X_tc_hrfed[vol_t, :]]

        print('hrf convolution: sample {0} to {1} finished'.format(
                batch, batches[k+1]))

    if np.ndim(X) == 1:
        X_hrfed = np.squeeze(X_hrfed)

    return X_hrfed
