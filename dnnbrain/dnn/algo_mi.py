import numpy as np
from skimage import segmentation 
from dnnbrain.dnn.algo import Algorithm


class MinmalParcelImage(Algorithm):
    """
    A class to generate minmal image for target channels from a DNN model 
    """
    def __init__(self, dnn, layer=None, channel=None, activaiton_criterion=None, search_criterion=None):
        """
        Parameter:
        ---------
        dnn[DNN]: dnnbrain's DNN object
        layer[str]: name of the layer where you focus on
        channel[int]: sequence number of the channel where you focus on
        activaiton_criterion[str]: the criterion of how to pooling activaiton
        search_criterion[str]: the criterion of how to search minimal image
        """
        super(MinmalParcelImage, self).__init__(dnn, layer, channel)
        self.parcel = None
       
    def set_params(self, activaiton_criterion='max', search_criterion='max'):
        """
        Set parameter for searching minmal image
        
        Parameter:
        ---------
        activaiton_criterion[str]: the criterion of how to pooling activaiton
        search_criterion[str]: the criterion of how to search minimal image
        """
        self.activaiton_criterion = activaiton_criterion
        self.search_criterion = search_criterion

    def _generate_decompose_parcel(self, image, segments):
        """
        Decompose image to multiple parcels using the given segments and
        put each parcel into a separated image with a black background
        
        Parameter:
        ---------
        image[ndarray]: shape (height,width,n_chn) 
        segments[ndarray]: shape (width, height).Integer mask indicating segment labels.
        
        Return:
        ---------
        parcel[ndarray]: shape (n_parcel,height,width,n_chn)
        """
        self.parcel = np.zeros((np.max(segments)+1,image.shape[0],image.shape[1],3),dtype=np.uint8)
        #generate parcel
        for label in np.unique(segments):
            self.parcel[label][segments == label] = image[segments == label]
        return self.parcel
        
    def felzenszwalb_decompose(self, image, scale=100, sigma=0.5, min_size=50):
        """
        Decompose image to multiple parcels using felzenszwalb method and
        put each parcel into a separated image with a black background
        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        
        Return:
        ---------
        parcel[ndarray]: shape (n_parcel,height,width,n_chn)
        """
        #decompose image
        segments = segmentation.felzenszwalb(image, scale, sigma, min_size)
        #generate parcel
        self.parcel = self._generate_decompose_parcel(image, segments)
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
        parcel[ndarray]: shape (n_parcel,height,width,n_chn)
        """
        #decompose image
        segments = segmentation.slic(image, n_segments, compactness, sigma)
        #generate parcel
        self.parcel = self._generate_decompose_parcel(image, segments)
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
        parcel[ndarray]: shape (n_parcel,height,width,n_chn)
        """
        #decompose image
        segments = segmentation.quickshift(image, kernel_size, max_dist, ratio)
        #generate parcel
        self.parcel = self._generate_decompose_parcel(image, segments)
        return self.parcel
    
    def sort_parcel(self, order='descending'):
        """
        sort the parcel according the activation of dnn. 
        
        Parameter:
        ---------
        order[str]: ascending or descending
        
        Return:
        ---------
        parcel[ndarray]: shape (n_parcel,height,width,n_chn) parcel after sorted
        """
        #change its shape(n_parcel,n_chn,height,width)
        parcel = self.parcel.transpose((0,3,1,2))
        #compute activation
        dnn_acts = self.dnn.compute_activation(parcel, self.mask).pool(self.activaiton_criterion).get(self.mask.layers[0])
        act_all = dnn_acts.flatten()
        #sort the activation in order
        if order == 'descending':
            self.parcel = self.parcel[np.argsort(-act_all)]
        else:
            self.parcel = self.parcel[np.argsort(act_all)]
                
        return self.parcel
        
    def combine_parcel(self, indices):
        """
        combine the indexed parcel into a image
        
        Parameter:
        ---------
        indices[int|list|slice]: subscript indices
        
        Return:
        -----
        image_container[ndarray]: shape (n_chn,height,width)
        """
        #compose parcel correaspond with indices
        if isinstance(indices, int):
            image_compose = np.sum(self.parcel[[indices]],axis=0)
        elif isinstance(indices, (list,slice)):
            image_compose = np.sum(self.parcel[indices],axis=0)
        return image_compose
    
    def generate_minmal_image(self):
        """
        Generate minimal image. We first sort the parcel by the activiton and 
        then iterate to find the combination of the parcels which can maximally
        activate the target channel.
        
        Note: before call this method, you should call xx_decompose method to 
        decompose the image into parcels. 
        
        Return:
        ---------
        image_min[ndarray]: final minimal images in shape (height,width,n_chn)
        """
        if self.parcel is None: 
            raise AssertionError('Please run decompose method to '
                                 'decompose the image into parcels')
        # sort the image
        self.sort_parcel()
        # iterater combine image to get activation
        parcel_add = np.zeros((self.parcel.shape[0],self.parcel.shape[1],self.parcel.shape[2],3),dtype=np.uint8)
        for index in range(self.parcel.shape[0]):
            parcel_mix = self.combine_parcel(slice(index+1))
            parcel_add[index] = parcel_mix[np.newaxis,:,:,:]
        # change its shape(n_parcel,n_chn,height,width) to fit dnn_activation
        parcel_add = parcel_add.transpose((0,3,1,2))
        # get activation
        dnn_act = self.dnn.compute_activation(parcel_add, self.mask).pool(self.activaiton_criterion).get(self.mask.layers[0])
        act_add = dnn_act.flatten()
        # generate minmal image according to the search_criterion
        if self.search_criterion == 'max':
            image_min = parcel_add[np.argmax(act_add)]
            image_min = np.squeeze(image_min).transpose(1,2,0)
        else:
            pass
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