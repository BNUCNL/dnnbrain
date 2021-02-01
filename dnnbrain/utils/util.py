import cv2
import random
import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau, zscore
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from dnnbrain.dnn.core import Mask


def get_frame_time_info(vid_file, original_onset, interval=1, before_vid=0, after_vid=0):
    """
    Extract frames of interest from a video with their onsets and durations,
    according to the experimental design.

    Parameters
    -----------
    vid_file : str 
        Video file path.
    original_onset : float 
        The first stimulus' time point relative to the beginning of the response.
        For example, if the response begins at 14 seconds after the first stimulus, 
        the original_onset is -14.
    interval : int 
        Get one frame per 'interval' frames,
    before_vid : float 
        Display the first frame as a static picture for 'before_vid' seconds before video.
    after_vid : float 
        Display the last frame as a static picture for 'after_vid' seconds after video.

    Returns
    --------
    frame_nums : list 
        Sequence numbers of the frames of interest.
    onsets : list 
        Onsets of the frames of interest.
    durations : list 
        Durations of the frames of interest.
    """
    assert isinstance(interval, int) and interval > 0, "Parameter 'interval' must be a positive integer!"

    # load video information
    vid_cap = cv2.VideoCapture(vid_file)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    n_frame = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # generate sequence numbers
    frame_nums = list(range(1, n_frame+1, interval))

    # generate durations
    duration = 1 / fps * interval
    durations = [duration] * len(frame_nums)
    durations[0] = durations[0] + before_vid
    durations[-1] = durations[-1] + after_vid

    # generate onsets
    onsets = [original_onset]
    for d in durations[:-1]:
        onsets.append(onsets[-1] + d)

    return frame_nums, onsets, durations


def gen_dmask(layers=None, channels='all', dmask_file=None):
    """
    Generate DNN mask object by:
    1. combining layers and channels.
    2. loading from dmask file.

    Parameters
    ----------
    layers : list 
        Layer names.
    channels : str, list 
        Channel numbers.
        It will be ignored if layers is None.
    dmask_file : str 
        A .dmask.csv file.

    Return
    ------
    dmask : Mask 
        DNN mask.
    """
    # set some assertions
    assert np.logical_xor(layers is None, dmask_file is None), \
        "Use one and only one of the 'layers' and 'dmask_file'!"

    dmask = Mask()
    if layers is None:
        # load from dmask file
        dmask.load(dmask_file)
    else:
        # combine layers and channels
        # contain all rows and columns for each layer
        n_layer = len(layers)
        if n_layer == 0:
            raise ValueError("'layers' can't be empty!")
        elif n_layer == 1:
            # All channels belong to the single layer
            dmask.set(layers[0], channels=channels)
        else:
            if channels == 'all':
                # contain all channels for each layer
                for layer in layers:
                    dmask.set(layer)
            elif n_layer == len(channels):
                # one-to-one correspondence between layers and channels
                for layer, chn in zip(layers, channels):
                    dmask.set(layer, channels=[chn])
            else:
                raise ValueError("channels must be 'all' or a list with same length as layers"
                                 " when the length of layers is larger than 1.")
    return dmask


def normalize(array):
    """
    Normalize an array's value domain to [0, 1]

    Parameter:
    ---------
    array : ndarray 
        A numpy array.

    Return:
    ------
    array : ndarray 
        A numpy array after normalization.
    """
    array = (array - array.min()) / (array.max() - array.min())

    return array


def topk_accuracy(pred_labels, true_labels, k):
    """
    Calculate top k accuracy for the classification results.

    Parameters:
    ----------
    pred_labels : array-like 
        Predicted labels, 2d array with shape as (n_stim, n_class).
        Each row's labels are sorted from large to small their probabilities.
    true_values : array-like 
        True values, 1d array with shape as (n_stim,).
    k : int
        The number of tops.

    Return:
    acc : float 
        Top k accuracy.
    """
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)
    assert pred_labels.shape[0] == true_labels.shape[0], 'The number of stimuli of pred_labels' \
                                                         ' and true_labels are mismatched.'
    assert 0 < k <= pred_labels.shape[1], 'k is out of range.'

    acc = 0.0
    for i in range(k):
        acc += np.sum(pred_labels[:, i] == true_labels)
    acc = acc / len(true_labels)

    return acc


def clustering(data, n_clusters, method, **kwargs):
    """
    Parameters
    ----------
    data : {array-like, sparse matrix} of shape (n_samples, n_features)
    n_clusters : int
        The number of clusters to form
        It will be ignored when the method is 'DBSCAN'.
    method : str
        specify the cluster method
    kwargs : keyword arguments

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    """
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, **kwargs)
        labels = kmeans.fit_predict(data)
    elif method == 'HAC':
        hac = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        labels = hac.fit_predict(data)
    elif method == 'DBSCAN':
        dbscan = DBSCAN(**kwargs)
        labels = dbscan.fit_predict(data)
    else:
        raise ValueError("not supported method")

    return labels


def permutation_RSA(rdm1, rdm2, corr_type='pearson', n_iter=10000):
    """
    Adapted from (Nili et al., 2014, PLOS Computational Biology)
    Test the relatedness of two RDMs by permutating item labels.

    Parameters
    ----------
    rdm1 : ndarray
        shape=(n_item, n_item)
    rdm2 : ndarray
        shape=(n_item, n_item)
    corr_type : str
        correlation measure to be used
        choices=('pearson', 'spearman', 'kendall')
        Default is 'pearson'.
    n_iter : int
        the number of iterations of permutation
        Default is 10000.

    Returns
    -------
    observed_R : float
        the correlation between two RDMs
    permuted_Rs : ndarray
        shape=(n_iter,)
        correlations between two RDMs during permutation
    P : float
        P-value of the observed correlation based on the distribution of permuted correlations.
    """
    assert rdm1.shape[0] == rdm2.shape[0], "The number of items is unmatched between two RDMs."
    n_item = rdm1.shape[0]

    # prepare correlation method
    if corr_type == 'spearman':
        corr = spearmanr
    elif corr_type == 'pearson':
        corr = pearsonr
    elif corr_type == 'kendall':
        corr = kendalltau
    else:
        raise ValueError("Correlation type should be one of ('pearson', 'spearman', 'kendall')")

    # calculate observed correlation
    triu_idx_arr = np.tri(n_item, k=-1, dtype=np.bool).T
    rdm1_vec = rdm1[triu_idx_arr]
    rdm2_vec = rdm2[triu_idx_arr]
    observed_R = corr(rdm1_vec, rdm2_vec)[0]

    # calculate correlation between two RDMs at each iteration.
    indices = list(range(n_item))
    permuted_Rs = np.ones(n_iter) * np.nan
    for iter_idx in range(n_iter):
        random.shuffle(indices)
        rdm1_vec = rdm1[indices][:, indices][triu_idx_arr]
        permuted_Rs[iter_idx] = corr(rdm1_vec, rdm2_vec)[0]

    # calculate p-value
    n1 = np.sum(permuted_Rs >= observed_R)
    n2 = np.sum(permuted_Rs <= observed_R)
    P = min(n1, n2) / n_iter

    return observed_R, permuted_Rs, P


def ceiling_RSA(rdms, corr_type='pearson'):
    """
    Adapted from (Nili et al., 2014, PLOS Computational Biology)
    Given a set of observed RDMs (e.g. from multiple subjects), this function estimates
    upper and lower bounds on the ceiling, i.e. the highest average RDM correlation
    (across the observed RDMs) that the true model's RDM prediction achieves given the variability
    of the estimates.

    Parameters
    ----------
    rdms : ndarray
        shape=(n_observation, n_item, n_item)
        observed RDMs
    corr_type : str
        Correlation measure to be used, choices=('pearson',).
        If is 'pearson', z-transforming the dissimilarities in each observed RDM before estimating.
        Because the estimating procedure involves averaging across observations.
        Default is 'pearson'.

    Returns
    -------
    lower_boundary : float
    upper_boundary : float
    best_fitted_rdm : ndarray
        shape=(n_item, n_item)
        the average RDM across observations after preprocessing according to corr_type.
    """
    n_observation, n_item = rdms.shape[0], rdms.shape[1]
    triu_idx_arr = np.tri(n_item, k=-1, dtype=np.bool).T
    rdm_vecs = rdms[:, triu_idx_arr]

    # prepare correlation
    if corr_type == 'pearson':
        rdm_vecs = zscore(rdm_vecs, 1)
        corr = pearsonr
    else:
        raise ValueError("Correlation type should be one of ('pearson',)")

    # estimate lower boundary by leave-one-out method
    LOO_corrs = np.ones(n_observation) * np.nan
    for i in range(n_observation):
        fitted_rdm_vec = np.mean(np.delete(rdm_vecs, i, 0), 0)
        LOO_corrs[i] = corr(fitted_rdm_vec, rdm_vecs[i])[0]
    lower_boundary = np.mean(LOO_corrs)

    # estimate upper boundary
    best_fitted_rdm_vec = np.mean(rdm_vecs, 0)
    upper_boundary = np.mean([corr(best_fitted_rdm_vec, rdm_vecs[i])[0] for i in range(n_observation)])
    best_fitted_rdm = np.zeros((n_item, n_item))
    best_fitted_rdm[triu_idx_arr] = best_fitted_rdm_vec
    best_fitted_rdm = best_fitted_rdm + best_fitted_rdm.T

    return lower_boundary, upper_boundary, best_fitted_rdm
