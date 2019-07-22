import numpy as np
from functools import partial

def extract_brain_activation(brainimg, mask, roilabels, method='mean'):
    """
    Extract brain activation from ROI
    
    Parameters:
    ------------
    brainimg[array]: A 4D brain image array with the first dimension correspond to pictures and the rest 3D correspond to brain images
    mask[array]: A 3D brain image array with the same size as the rest 3D of brainimg.
    roilabels[list/array]: ROI labels
    method[str]: method to integrate activation from each ROI, by default is 'mean'.
    
    Returns:
    ---------
    roisignals[array]: extracted brain activation
    """
    if method == 'mean':
        calc_way = partial(np.mean, axis=0)
    elif method == 'std':
        calc_way = partial(np.std, axis=0)
    elif method == 'max':
        calc_way = partial(np.max, axis=0)
    elif method == 'voxel':
        calc_way = np.array
    else:
        raise Exception('We haven''t support this method, please contact authors to implement.')
    
    assert brainimg.shape[1:] == mask.shape, "brainimg and mask are mismatched."
    roisignals = []    
    for i, lbl in enumerate(roilabels):
        act_idx = np.transpose(np.where(mask==lbl))
        roisignal_tmp = [brainimg[:, act_idx[j][0], act_idx[j][1], act_idx[j][2]] for j in range(len(act_idx))]
        roisignals.append(calc_way(roisignal_tmp))
    roisignals = np.array(roisignals)
    return roisignals