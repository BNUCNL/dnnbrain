import numpy as np
from dnnbrain.dnn import io as dio
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
    dnnact[numpy.array]: DNN activation, A 3D array with its shape as (n_picture, n_channel, n_column)
    """
    assert (fe_axis is None) == (fe_meth is None), 'Please specify fe_axis and fe_meth at the same time.'

    # get dnn activation
    loader = dio.NetLoader(netname)
    actmodel = dnn_truncate(loader, layer)
    actmodel.eval()
    dnnact = []
    count = 0  # count the progress
    for picdata, _ in input:
        dnnact_part = actmodel(picdata)
        dnnact.extend(dnnact_part.detach().numpy())
        count += dnnact_part.shape[0]
        print('Extracted acts:', count)
    dnnact = np.array(dnnact)
    dnnact = dnnact.reshape((dnnact.shape[0], dnnact.shape[1], -1))

    # mask the data
    if channel is not None:
        dnnact = dnnact[:, channel, :]
    if column is not None: 
        dnnact = dnnact[:, :, column]
        
    # feature extraction
    if fe_axis is not None:
        fe_meths = {
            'max': np.max,
            'mean': np.mean,
            'median': np.median
        }

        if fe_axis == 'layer':
            dnnact = dnnact.reshape((dnnact.shape[0], -1))
            dnnact = fe_meths[fe_meth](dnnact, -1)[:, np.newaxis, np.newaxis]
        elif fe_axis == 'channel':
            dnnact = fe_meths[fe_meth](dnnact, 1)[:, np.newaxis, :]
        elif fe_axis == 'column':
            dnnact = fe_meths[fe_meth](dnnact, 2)[:, :, np.newaxis]
        else:
            raise ValueError('fe_axis should be layer, channel or column')
    
    return dnnact


def convolve_hrf(X, onsets, durations, n_vol, tr, ops=100):
    """
    Convolve each X's column iteratively with HRF and align with the timeline of BOLD signal

    parameters:
    ----------
    X[array]: [n_event, n_sample]
    onsets[array_like]: in sec. size = n_event
    durations[array_like]: in sec. size = n_event
    n_vol[int]: the number of volumes of BOLD signal
    tr[float]: repeat time in second
    ops[int]: oversampling number per second

    Returns:
    ---------
    X_hrfed[array]: the result after convolution and alignment
    """
    assert np.ndim(X) == 2, 'X must be a 2D array'
    assert X.shape[0] == len(onsets) and X.shape[0] == len(durations), 'The length of onsets and durations should ' \
                                                                       'be matched with the number of events.'
    assert ops in (10, 100, 1000), 'Oversampling rate must be one of the (10, 100, 1000)!'

    # unify the precision
    decimals = int(np.log10(ops))
    onsets = np.round(np.asarray(onsets), decimals=decimals)
    durations = np.round(np.asarray(durations), decimals=decimals)
    tr = np.round(tr, decimals=decimals)

    n_clipped = 0  # the number of clipped time points earlier than the start point of response
    onset_min = onsets.min()
    if onset_min > 0:
        print("The earliest event's onset is later than the start point of response.\n"
              "We supplement it with zero-value event to align with the response.")
        X = np.insert(X, 0, np.zeros(X.shape[1]), 0)
        onsets = np.insert(onsets, 0, 0, 0)
        durations = np.insert(durations, 0, onset_min, 0)
        onset_min = 0
    elif onset_min < 0:
        print("The earliest event's onset is earlier than the start point of response.\n"
              "We clip the earlier time points after hrf_convolution to align with the response.")
        n_clipped = int(-onset_min * ops)

    # do convolution in batches for trade-off between speed and memory
    batch_size = int(100000 / ops)
    bat_indices = np.arange(0, X.shape[-1], batch_size)
    bat_indices = np.r_[bat_indices, X.shape[-1]]

    vol_t = (np.arange(n_vol) * tr * ops).astype(int)  # compute volume acquisition timing
    n_time_point = int(((onsets + durations).max()-onset_min) * ops)
    X_hrfed = np.zeros([n_vol, 0])
    for idx, bat_idx in enumerate(bat_indices[:-1]):
        X_bat = X[:, bat_idx:bat_indices[idx+1]]
        # generate X raw time course
        X_tc = np.zeros((n_time_point, X_bat.shape[-1]), dtype=np.float32)
        for i, onset in enumerate(onsets):
            onset_start = int(onset * ops)
            onset_end = int(onset_start + durations[i] * ops)
            X_tc[onset_start:onset_end, :] = X_bat[i, :]

        # generate hrf kernel
        hrf = spm_hrf(tr, oversampling=tr*ops)
        hrf = hrf[:, np.newaxis]

        # convolve X raw time course with hrf kernal
        X_tc_hrfed = convolve(X_tc, hrf, method='fft')
        X_tc_hrfed = X_tc_hrfed[n_clipped:, :]

        # downsample to volume timing
        X_hrfed = np.c_[X_hrfed, X_tc_hrfed[vol_t, :]]

        print('hrf convolution: sample {0} to {1} finished'.format(bat_idx, bat_indices[idx+1]))

    return X_hrfed
