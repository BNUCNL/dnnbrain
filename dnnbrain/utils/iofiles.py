try:
    import torchvision
    import numpy as np
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    print('Please install pytorch and torchvision in your work station')
    
try:
    import nibabel as nib
    import cifti
except ModuleNotFoundError:
    print('Please install nibabel and cifti in your work station')

class _ImageFolder(datasets.ImageFolder):
    """
    Reconstruct ImageFolder to allow it output picture name.
    """
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        picname = path.split('/')[-1]
        # In case running code in windows
        picname = picname.split('\\')[-1]
        return img, target, picname

class ImgLoader():
    def __init__(self, imgpath):
        """
        """
        self.imgpath = imgpath
        
    def gen_dataloader(self, imgcropsize, transform = None, batch_size = 8, shuffle=True, num_workers=1):
        """
        Generate dataloader from image path
		
        Parameters:
        ------------
        imgpath[str]: path of stimuli picture
        imgresize[int/list]: resize images to make it suitable to input of a specific network
        transform[transform.Compose]: transformation ways
        batch_size[int]: batch size
        shuffle[bool]: shuffle images or not
        num_workers[int]: cpu workers used in model trainning
        
        Returns:
        ---------
        dataloader[dataloader instance]: dataloader which could be used directly in CNN models
        """
        if transform is None:
            transform = transforms.Compose([
                            transforms.Resize(imgcropsize),
                            transforms.ToTensor()
                                           ])
        pak_datasets = _ImageFolder(self.imgpath, transform)
        dataloader = DataLoader(pak_datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.dataloader = dataloader
        return dataloader
        
        
class BrainImgLoader():
    def __init__(self, imgpath):
        """
        """
        self.imgpath = imgpath
        
    def load_brainimg(self):
        """
        Load brain image identified by its suffix
        suffix now support
        
        Nifti: .nii.gz
        freesurfer: .mgz, .mgh
        cifti: .dscalar.nii, .dlabel.nii, .dtseries.nii
        
        Parameters:
        ------------
        
        Returns:
        ------------
        brain_img[np.array]: data of brain image
        """
        imgname = self.imgpath.split('/')[-1]
        # In case running code in windows
        imgname = imgname.split('\\')[-1]
        imgsuffix = imgname.split('.')[1:]
        imgsuffix = '.'.join(imgsuffix)
        
        if imgsuffix == 'nii.gz':
            brain_img = nib.load(self.imgpath).get_data()
        elif imgsuffix == 'mgz' or imgsuffix == 'mgh':
            brain_img = nib.freesurfer.load(self.imgpath)
        elif imgsuffix == 'dscalar.nii' or imgsuffix == 'dlabel.nii' or imgsuffix == 'dtseries.nii':
            brain_img, _ = cifti.read(self.imgpath)
        else:
            raise Exception('Not support this format of brain image data, please contact with Taicheng Huang to update this function.')
        return brain_img
        
        
        
def save_activation(activation,outpath,net,layer,channel=None):
    """
    Save activaiton data to a csv file in outpath

    Parameteers:
    ------------
    activation[4darray]: sitmulus x channel x pixel x pixel
    outpath[str]:outpath and outfilename
    net[str]:neuron network name
    layer[int]:the number of layer of network
    channel[list]: the number list of channel/filter

    """
    activation2d = np.reshape(activation,(np.prod(activation.shape[0:2]),-1,),order='C')
    channelline = np.array([channel+1 for channel in range(activation.shape[1])]*activation.shape[0])
    stimline = []
    for i in range(activation.shape[0]):
        a = [i + 1 for j in range(activation.shape[1])]
        stimline = stimline + a
    stimline = np.array(stimline)
    channelline = np.reshape(channelline, (channelline.shape[0], 1))
    stimline = np.reshape(stimline, (stimline.shape[0], 1))
    activation2d = np.concatenate((stimline,channelline,activation2d),axis=1)
    if channel is None:
        np.savetxt('{}_{}_{}.csv'.format(outpath,net,layer),activation2d,delimiter = ',')
    else:
        np.savetxt('{}{}_{}_{}.csv'.format(outpath, net,layer,channel), activation2d, delimiter=',')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

