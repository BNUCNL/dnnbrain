import numpy as np
from abc import ABC, abstractmethod
from skimage import segmentation 
from dnnbrain.dnn.core import Mask, Algorithm


class MinmalParcelImage(Algorithm):
    """
    A class to generate minmal image for target channels from a DNN model 
   
    """
    def __init__(self, dnn, layer=None, channel=None):
       
       super(MinmalParcelImage, self).__init__(dnn, layer, channel)
       self.parcel = None
       
    def set_params(self, activaiton_criterion, search_criterion='max'):
        """
        Set parameter for searching minmal image
        criterion: criterion to 
        """
        self.activaiton_criterion = activaiton_criterion
        self.search_criterion = search_criterion

    def felzenszwalb_decompose(self, image, scale=100, sigma=0.5, min_size=50):
        """
        Decompose image to multiple parcels using felzenszwalb method and
        put each parcel into a separated image with a black background
        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        
        Return:
        ---------
        parcel[list]: shape (n_parcel,n_chn,height,width) all patches of ndarray 
        """
        self.parcel = []
        segments = segmentation.felzenszwalb(image, scale, sigma, min_size)
        #generate parcel
        for label in np.unique(segments):
            #Create a black backgroud
            image_bkg = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
            image_bkg[segments == label] = image[segments == label]
            image_bkg.transpose((2,0,1))
            self.parcel.append(image_bkg)
            
        return self.parcel

    def slic_decompose(self, image, n_segments=250, compactness=10, sigma=1):
        """
        Decompose image to multiple parcels using slic method and
        put each parcel into a separated image with a black background        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        meth[str]: method to decompose images
        
        Return:
        ---------
        parcel[list]: shape (n_parcel,n_chn,height,width) all patches of ndarray 
        """
        self.parcel = []
        segments = segmentation.slic(image, n_segments, compactness, sigma)
        #generate parcel
        for label in np.unique(segments):
            #Create a black backgroud
            image_bkg = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
            image_bkg[segments == label] = image[segments == label]
            image_bkg.transpose((2,0,1))
            self.parcel.append(image_bkg)
            
        return self.parcel

    def quickshift_decompose(self, image, kernel_size=3, max_dist=6, ratio=0.5):
        """
        Decompose image to multiple parcels using quickshift method and
        put each parcel into a separated image with a black background
        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        meth[str]: method to decompose images
        
        Return:
        ---------
        parcel[list]: shape (n_parcel,n_chn,height,width) all patches of ndarray 
        """
        self.parcel = []
        segments = segmentation.quickshift(image, kernel_size, max_dist, ratio)
        #generate parcel
        for label in np.unique(segments):
            #Create a black backgroud
            image_bkg = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
            image_bkg[segments == label] = image[segments == label]
            image_bkg.transpose((2,0,1))
            self.parcel.append(image_bkg)
            
        return self.parcel
    
    def sort_parcel(self, order='descending'):
        """
        sort the parcel according the activation of dnn. 
        
        Parameter:
        ---------
        order[str]: ascending or descending
        
        Return:
        ---------
        parcel[list]: shape (n_parcel,n_chn,height,width) 
        """
        parcel = np.asarray(self.parcel)
        #compute activation
        dnn_acts = self.dnn.compute_activation(parcel, self.dmask).pool('max').get(self.layer)
        act_all = dnn_acts.flatten()
        #sort the activation in descending
        self.parcel = self.parcel(np.argsort(-act_all))
        
        return self.parcel
        
    def combine_parcel(self, index):
        """
        combine the indexed parcel into a image
        
        Parameter:
        ---------
        index[int]: the index that you want to combine 
        
        Return:
        ------
        image_container[ndarray]: shape (n_chn,height,width)
        """
        #initialize image_container
        parcel = np.asarray(self.parcel)
        image_container = np.zeros((parcel.shape[1],parcel.shape[2],3),dtype=np.uint8)  
        #loop to generate combine_parcel with targeted index
        for pic in range(index):
            image_container += self.parcel[pic] 
        
        return image_container
    
    def generate_minmal_image(self):
        """
        Generate minimal image. We first sort the parcel by the activiton and 
        then iterate to find the combination of the parcels which can maximally
        activate the target channel.
        
        Note: before call this method, you should call xx_decompose method to 
        decompose the image into parcels. 
        
        Return:
        ---------
        image_min[ndarray]: fininal minimal images in shape (n_chn,height,width)
        """
        if self.parcel is None: 
            raise AssertionError('Please run decompose method to '
                                 'decompose the image into parcels')
        # workflow
        # sort the image
        # iterater combine image to get activation
        self.sort_parcel()
        parcel_add = []
        for index in self.parcel.shape[0]:
            parcel_index = self.combine_parcel(index)
            parcel_add.append(parcel_index)
        parcel_add = np.asarray(parcel_add)
        # return the opimized curve and minmal image
        dnn_act = self.dnn.compute_activation(parcel_add, self.dmask).pool('max').get(self.layer)
        act_add = dnn_act.flatten()
        image_min = self.combine_parcel(np.argmax(act_add))
        
        return image_min
        
class MinmalComponentImage(Algorithm):
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """

    def set_params(self,  meth='pca', criterion='max'):
        """Set parameter for the estimator"""
        self.meth = meth
        self.criterion = criterion
        
        
    def pca_decompose(self):
        pass
    
    
    def ica_decompose(self):
        pass
    
    
    
    def sort_componet(self, order='descending'):
        """
        sort the component according the activation of dnn. 
        order[str]: ascending or descending
        """
      
    def combine_component(self, index):
        """combine the indexed component into a image"""
        pass 
    
    
    
    def generate_minmal_image(self):
        """
        Generate minimal image. We first sort the component by the activiton and 
        then iterate to find the combination of the components which can maximally
        activate the target channel.
        
        Note: before call this method, you should call xx_decompose method to 
        decompose the image into parcels. 
        
        Parameter:
        ---------
        stim[Stimulus]: stimulus
        Return:
    
        ------
        """