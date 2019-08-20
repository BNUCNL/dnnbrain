import pytest
import os
from dnnbrain.brain import io as brain_io

def test_load_brainimg(dnnbrain_path):
    """
    Test output of function: load_brainimg
    """
    parpath = os.path.join(dnnbrain_path, 'data', 'test_data', 'mri_images')
    # Test nifti activation loading
    nifti_act, _ = brain_io.load_brainimg(os.path.join(parpath, 'test_actdata.nii.gz'), ismask=False)
    assert nifti_act.ndim == 4
    assert nifti_act.shape == (2,91,109,91)
    # Test nifti mask loading
    nifti_mask, _ = brain_io.load_brainimg(os.path.join(parpath, 'test_mask.nii.gz'), ismask=True)
    assert nifti_mask.ndim == 3
    assert nifti_mask.shape == (91,109,91)
    # Test freesurfer mgz files loading
    mgz_mask, _ = brain_io.load_brainimg(os.path.join(parpath, 'test_mask.mgz'), ismask=True)
    assert mgz_mask.ndim == 3
    assert mgz_mask.shape == (163842,1,1)
    # Test gifti files loading
    gii_act, _ = brain_io.load_brainimg(os.path.join(parpath, 'test_data.shape.gii'), ismask=False)
    assert gii_act.ndim == 4
    assert gii_act.shape == (1,32492,1,1)
    with pytest.raises(AssertionError):
        gii_act, _ = brain_io.load_brainimg(os.path.join(parpath, 'test_data.surf.gii'), ismask=False)
    # Test cifti files loading
    cifti_mask, _ = brain_io.load_brainimg(os.path.join(parpath, 'test_mask.dscalar.nii'), ismask=True)
    assert cifti_mask.ndim == 3
    assert cifti_mask.shape == (1,59412,1)
    

def test_extract_brain_activation(dnnbrain_path):
    """
    Test output of function: extract_brain_activation
    """
    parpath = os.path.join(dnnbrain_path, 'data', 'test_data', 'mri_images')
    nifti_act, _ = brain_io.load_brainimg(os.path.join(parpath, 'test_actdata.nii.gz'), ismask=False)
    nifti_mask, _ = brain_io.load_brainimg(os.path.join(parpath, 'test_mask.nii.gz'), ismask=True)
    masksize = nifti_mask[nifti_mask==1].shape[0]
    roisignals_mean = brain_io.extract_brain_activation(nifti_act, nifti_mask, [1], method='mean')
    assert roisignals_mean[0].shape == (nifti_act.shape[0],)
    roisignals_voxel = brain_io.extract_brain_activation(nifti_act, nifti_mask, [1], method='voxel')
    assert roisignals_voxel[0].shape == (masksize, nifti_act.shape[0])
    
    
    
    
    
    
    