import numpy as np
from dnnbrain.dnn import io as iofiles
from dnnbrain.dnn.models import dnn_truncate
from nipy.modalities.fmri.hemodynamic_models import spm_hrf
from scipy.signal import convolve, resample


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


def generate_bold_regressor(X, onset, duration, vol_num, tr):
    """
    convolve event-format X with hrf and align with timeline of BOLD signal

    parameters:
    ----------
    X[array]: [n_event] or [n_event,n_sample]
    onset[list or array]: in sec. size = n_event
    duration[list or array]: list or array. in sec. size = n_event
    vol_num[int]: total volume number of BOLD signal
    tr[float]: in sec

    Returns:
    ---------
    X_hrfed[array]: same shape with X
    """

    onset = np.round(np.asarray(onset), decimals=3)
    duration = np.round(np.asarray(duration), decimals=3)
    tr = np.round(tr, decimals=3)

    if np.ndim(X) == 1:
        X = X[:, np.newaxis]

    batch_size = 1000
    batches = np.arange(0, X.shape[-1], batch_size)
    batches = np.r_[batches, X.shape[-1]]

    # compute volume acqusition timing
    vol_t = (np.arange(vol_num) * tr * 1000).astype(int)

    X_hrfed = np.zeros([vol_num, 0])
    for k, batch in enumerate(batches[:-1]):
        X_i = X[:, batch:batches[k+1]]
        # generate X raw time course in ms
        X_tc = np.zeros(
                [int((onset+duration).max()*1000), X_i.shape[-1]],
                dtype=np.float32)
        for i, onset_i in enumerate(onset):
            onset_i_start = int(onset_i*1000)
            onset_i_end = int(onset_i_start + duration[i]*1000)
            X_tc[onset_i_start:onset_i_end, :] = X_i[i, :]

        # generate hrf kernel
        hrf = spm_hrf(tr, oversampling=tr*1000, time_length=32, onset=0)
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
