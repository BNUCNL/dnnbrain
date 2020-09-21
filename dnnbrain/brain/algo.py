import numpy as np

from scipy.signal import convolve
from nipy.modalities.fmri.hemodynamic_models import spm_hrf


def convolve_hrf(X, onsets, durations, n_vol, tr, ops=100):
    """
    Convolve each X's column iteratively with HRF and align with the timeline of BOLD signal

    Parameters
    ----------
    X : array
        Shape = (n_event, n_sample)
    onsets : array_like
        In sec. size = n_event
    durations : array_like
        In sec. size = n_event
    n_vol : int
        The number of volumes of BOLD signal
    tr : float
        Repeat time in second
    ops : int
        Oversampling number per second
    
    Returns
    -------
    X_hrfed : array
        The result after convolution and alignment
    """
    assert np.ndim(X) == 2, 'X must be a 2D array'
    assert X.shape[0] == len(onsets) and X.shape[0] == len(durations), \
        'The length of onsets and durations should be matched with the number of events.'
    assert ops in (10, 100, 1000), 'Oversampling rate must be one of the (10, 100, 1000)!'

    # unify the precision
    decimals = int(np.log10(ops))
    onsets = np.round(np.asarray(onsets), decimals=decimals)
    durations = np.round(np.asarray(durations), decimals=decimals)
    tr = np.round(tr, decimals=decimals)

    n_clipped = 0  # the number of clipped time points earlier than the start point of response
    onset_min = onsets.min()
    if onset_min > 0:
        # The earliest event's onset is later than the start point of response.
        # We supplement it with zero-value event to align with the response.
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

        print('hrf convolution: sample {0} to {1} finished'.format(bat_idx+1, bat_indices[idx+1]))

    return X_hrfed
