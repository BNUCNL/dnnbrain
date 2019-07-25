import numpy as np
import os
from functools import partial
try:
    import nibabel as nib
    import cifti
except ModuleNotFoundError:
    raise Exception('Please install nibabel and cifti in your work station')

def load_brainimg(imgpath, ismask=False):
    """
    Load brain image identified by its suffix
    suffix now support
      
    Nifti: .nii.gz
    freesurfer: .mgz, .mgh
    cifti: .dscalar.nii, .dlabel.nii, .dtseries.nii
        
    Parameters:
    ------------
    imgpath: brain image data path
        
    Returns:
    ------------
    brain_img[np.array]: data of brain image
    header[header]: header of brain image
    """
    imgname = os.path.basename(imgpath)
    imgsuffix = imgname.split('.')[1:]
    imgsuffix = '.'.join(imgsuffix)

    if imgsuffix == 'nii.gz':
        brain_img = nib.load(imgpath).get_data()
        if not ismask:
            brain_img = np.transpose(brain_img,(3,0,1,2))
        header = nib.load(imgpath).header
    elif imgsuffix == 'mgz' or imgsuffix == 'mgh':
        brain_img = nib.freesurfer.load(imgpath).get_data()
        if not ismask:
            brain_img = np.transpose(brain_img, (3,0,1,2))
        header = nib.freesurfer.load(imgpath).header
    elif imgsuffix == 'dscalar.nii' or imgsuffix == 'dlabel.nii' or imgsuffix == 'dtseries.nii':
        brain_img, header = cifti.read(imgpath)
        if not ismask:
            brain_img = brain_img[...,None,None]
        else:
            brain_img = brain_img[...,None]
    else:
        raise Exception('Not support this format of brain image data, please contact with author to update this function.')
    return brain_img, header
    
    
def save_brainimg(imgpath, data, header):
    """
    Save brain image identified by its suffix
    suffix now support
     
    Nifti: .nii.gz
    freesurfer: .mgz, .mgh
    cifti: .dscalar.nii, .dlabel.nii, .dtseries.nii
        
    Parameters:
    ------------
    imgpath: brain image path to be saved
    data: brain image data matrix
    header: brain image header
        
    Returns:
    --------
    """
    imgname = os.path.basename(imgpath)
    imgsuffix = imgname.split('.')[1:]
    imgsuffix = '.'.join(imgsuffix)
    
    if imgsuffix == 'nii.gz':
        data = np.transpose(data,(1,2,3,0))
        outimg = nib.Nifti1Image(data, None, header)
        nib.save(outimg, imgpath)
    elif imgsuffix == 'mgz' or imgsuffix == 'mgh':
        data = np.transpose(data, (1,2,3,0))
        outimg = nib.MGHImage(data, None, header)
        nib.save(outimg, imgpath)
    elif imgsuffix == 'dscalar.nii' or imgsuffix == 'dlabel.nii' or imgsuffix == 'dtseries.nii':
        data = data[...,0,0]
        map_name = ['']*data.shape[0]
        bm_full = header[1]
        cifti.write(imgpath, data, (cifti.Scalar.from_names(map_names), bm_full))
    else:
        raise Exception('Not support this format of brain image data, please contact with author to update this function.')   


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