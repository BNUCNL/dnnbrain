import numpy as np

from dnnbrain.brain.algo import convolve_hrf


def test_convolve_hrf():

    # prepare
    X = np.random.randn(10, 2)
    onsets = np.arange(10)
    durations = np.ones(10)
    n_vol = 5
    tr = 2

    # do convolution
    X_hrfed = convolve_hrf(X, onsets, durations, n_vol, tr)

    # assert
    assert X_hrfed.shape == (n_vol, X.shape[1])
    np.testing.assert_almost_equal(X_hrfed[0], np.zeros(X.shape[1]))
