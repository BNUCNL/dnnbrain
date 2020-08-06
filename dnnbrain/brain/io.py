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
    Load brain image identified by its suffix.
    The supporting suffixes are as follows:
      
    Nifti: .nii.gz
    
    freesurfer: .mgz, .mgh
    
    gifti: .func.gii, .shape.gii
    
    cifti: .dscalar.nii, .dlabel.nii, .dtseries.nii
        
    Parameters
    ----------
    imgpath : str
        Brain image data path
        
    Returns
    -------
    brain_img : array
        Data of brain image
    header : header
        Header of brain image
    """
    imgname = os.path.basename(imgpath)

    if imgname.endswith('dscalar.nii') or imgname.endswith('dlabel.nii') or imgname.endswith('dtseries.nii'):
        brain_img, header = cifti.read(imgpath)
        if not ismask:
            brain_img = brain_img[...,None,None]
        else:
            brain_img = brain_img[...,None]
    elif ('nii.gz' in imgname) or (imgname.split('.')[-1]=='nii'):
        brain_img = nib.load(imgpath).get_data()
        if not ismask:
            brain_img = np.transpose(brain_img,(3,0,1,2))
        header = nib.load(imgpath).header
    elif imgname.endswith('mgz') or imgname.endswith('mgh'):
        brain_img = nib.freesurfer.load(imgpath).get_data()
        if not ismask:
            if brain_img.ndim == 3:
                brain_img = brain_img[...,None]
            brain_img = np.transpose(brain_img, (3,0,1,2))
        header = nib.freesurfer.load(imgpath).header
    elif imgname.endswith('gii'):
        assert not imgname.endswith('surf.gii'), "surf.gii is a geometry file, not an array activation."
        brain_img = nib.load(imgpath).darrays[0].data
        if not ismask:
            brain_img = brain_img[None,:,None,None]
        else:
            brain_img = brain_img[None,:,None]
        header = nib.load(imgpath).header
    else:
        raise Exception('Not support this format of brain image data, please contact with author to update this function.')
    return brain_img, header
    
    
def save_brainimg(imgpath, data, header):
    """
    Save brain image identified by its suffix.
    The supporting suffixes are as follows:
     
    Nifti: .nii.gz
    
    freesurfer: .mgz, .mgh
    
    cifti: .dscalar.nii, .dlabel.nii, .dtseries.nii
       
    Note that due to ways to store gifti image are differ from other images, 
    we didn't support to save data as a gifti image.
        
    Parameters
    ----------
    imgpath : str
        Brain image path to be saved
    data : ndarray
        Brain image data matrix
    header : header
        Brain image header
    """
    imgname = os.path.basename(imgpath)
    imgsuffix = imgname.split('.')[1:]
    assert len(imgsuffix)<4, "Please rename your brain image file for too many . in your filename."
    imgsuffix = '.'.join(imgsuffix)
    
    if imgsuffix == 'nii.gz':
        data = np.transpose(data, (1, 2, 3, 0))
        outimg = nib.Nifti1Image(data, None, header)
        nib.save(outimg, imgpath)
    elif imgsuffix == 'mgz' or imgsuffix == 'mgh':
        data = np.transpose(data, (1, 2, 3, 0))
        outimg = nib.MGHImage(data, None, header)
        nib.save(outimg, imgpath)
    elif imgsuffix == 'dscalar.nii' or imgsuffix == 'dlabel.nii' or imgsuffix == 'dtseries.nii':
        data = data[..., 0, 0]
        map_name = ['']*data.shape[0]
        bm_full = header[1]
        cifti.write(imgpath, data, (cifti.Scalar.from_names(map_name), bm_full))
    else:
        raise Exception('Not support this format of brain image data, please contact with author to update this function.')   


def extract_brain_activation(brainimg, mask, roilabels, method='mean'):
    """
    Extract brain activation from ROI.
    
    Parameters
    ----------
    brainimg : array
        A 4D brain image array with the first dimension correspond to pictures and the rest 3D correspond to brain images
    mask : array
        A 3D brain image array with the same size as the rest 3D of brainimg.
    roilabels : list, array
        ROI labels
    method : str
        Method to integrate activation from each ROI, by default is 'mean'.
    
    Returns
    -------
    roisignals : list
        Extracted brain activation. 
        Each element in the list is the extracted activation of the roilabels.
        Due to different label may contain different number of activation voxels, 
        the output activation could not stored as numpy array list.
    """
    if method == 'mean':
        calc_way = partial(np.mean, axis=1)
    elif method == 'std':
        calc_way = partial(np.std, axis=1)
    elif method == 'max':
        calc_way = partial(np.max, axis=1)
    elif method == 'voxel':
        calc_way = np.array
    else:
        raise Exception('We haven''t support this method, please contact authors to implement.')
    
    assert brainimg.shape[1:] == mask.shape, "brainimg and mask are mismatched."
    roisignals = []    
    for i, lbl in enumerate(roilabels):
        roisignals.append(calc_way(brainimg[:, mask==lbl]))
    return roisignals