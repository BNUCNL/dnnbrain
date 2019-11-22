import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from os.path import join as pjoin
from skimage.segmentation import felzenszwalb, slic, quickshift
from dnnbrain.dnn.core import Mask

class Algorithm(ABC):
    """ 
    An Abstract Base Classes class to define interface for dnn algorithm 
    """
    def __init__(self, dnn, layer=None, channel=None):
        """
        Parameter:
        ---------
        dnn[str]: The name of DNN net.
                  You should open dnnbrain.dnn.models
                  to check if the DNN net is supported.
        layer[str]: The name of layer in DNN net.
        channel[int]: The channel of layer which you focus on.
        """
        self.dnn = dnn
        self.layer = layer
        self.channel = channel
        self.dmask = Mask()
        self.dmask.set(self.layer, [self.channel, ])

    def set_layer(self, layer, channel):
        """
        Parameter:
        ---------
        layer[str]: The name of layer in DNN net.
        channel[int]: The channel of layer which you focus on.
        """
        self.layer = layer
        self.channel = channel
        self.dmask = Mask()
        self.dmask.set(self.layer, [self.channel, ])
        
    @abstractmethod
    def set_params(self):
        """ set parames """
        
    @abstractmethod
    def compute(self, image): 
        """Please implement your algorithm here"""
        
        
class MinmalParcelImage(Algorithm):
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
   
    """
    
    def set_params(self, meth='SLIC', criterion='max'):
        """Set parameter for the estimator"""
        self.meth  = meth
        self.criterion = criterion

    def decompose(self, image, meth):
        """
        decompose images to several segments
        
        Parameter:
        ---------
        image[ndarray] : shape (height,width,n_chn) 
        meth[str]: method to decompose images
        
        Return:
        ------
        segments[ndarray]: Integer mask indicating segment labels.
        """       
        if meth == 'felzenszwalb':
            segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
        if meth == 'SLIC':
            segments = slic(image, n_segments=250, compactness=10, sigma=1)
        if meth == 'quickshift':
            segments = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
    
        return segments
    
    def image_backgroud(self, image, RGB):
        """
        Generate a copy of image with same size but replace RGB
        
        Parameter:
        ---------
        image[ndarray] : (height,width,n_chn) shape
        RGB[tuple]: RGB of replace color shape (R,G,B)
        
        Return:
        ------
        image_copy[ndarray]: a copy of image with same size but replace RGB.
        """  
        #get a copy of the image with same size
        image_copy=image.copy() 
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image_copy[i,j] = RGB
                
        return image_copy

    def generate_patch_single(self, image, segments, RGB=(255,255,255)):
        """
        Generate all patches of an image,the other area is replaced by image_backgroud
        
        Parameter:
        ---------
        image[ndarray] : (height,width,n_chn) shape
        segments[ndarray]: label of segmented image
        RGB[tuple]: RGB of replace color
        
        Return:
        ------
        patch_all[ndarray]: its shape is (n_segments, n_chn, height, width) 
        """         
        patch_all = []
        #find the location and replace other area with targeted RGB
        for label in np.unique(segments):
            image_RGB = self.image_backgroud(image, RGB)
            place_x,place_y = np.where(segments==label)  
            for i in range(len(place_x)):
                image_RGB[place_x[i],place_y[i]]=image[place_x[i],place_y[i]] 
            
            #chage the shape(height,width,chn) to (chn,height,width)
            patch = image_RGB.transpose((2,0,1)) 
            
            #append the ndarray in a list
            patch_all.append(patch)
        
        #translate list to ndarray
        patch_all = np.asarray(patch_all)
        
        return patch_all

    
    def generate_patch_mix(self, image, segments, act_sorted, RGB=(255,255,255)):
        """
        Generate all patches of an image,the sequence is to put the patch of activation from max to min
        the other area is replaced by image_backgroud
        
        Parameter:
        ---------
        image[ndarray] : (height,width,n_chn) shape
        segments[ndarray]: label of segmented image 
        act_sorted[list] :index of activation from max to min in the original label
        RGB[tuple]: RGB of replace color
        
        Return:
        ------
        patch_all[ndarray]: its shape is (n_segments, n_chn, height, width)
        """    
        image_RGB = self.image_backgroud(image, RGB)
        
        #get size in order to fit dnn.compute_activation
        size = [1,image.shape[2],image.shape[0],image.shape[1]]
        #create a zeros ndarray to ensure concatenate can be down
        patch_add=np.zeros(size, dtype='uint8')
        
        #find the location and replace other area with noise
        for index in act_sorted:
            place_x,place_y = np.where(segments==index)  
            for i in range(len(place_x)):
                image_RGB[place_x[i], place_y[i]]=image[place_x[i], place_y[i]] 
            
            #change the shape(height,width,chn) to (chn,height,width) and add a newaxis
            patch = image_RGB.transpose((2, 0, 1))[np.newaxis,:,:,:]
            
            #concatenate patch into patch_add
            patch_add = np.concatenate((patch_add,patch),axis=0)
            
        #delete the original zeros ndarray
        patch_add = np.delete(patch_add,0,0)
        
        return patch_add
           
        
    def compute(self, stim, RGB=(255,255,255)):
        """
        Generate minimal image for image listed in stim object
        
        Parameter:
        ---------
        stim[Stimulus]: stimulus

        Return:
        ------
        image_min_all[list]: all minimal images
        """      
        #prepare a list to preserve minimal image
        image_min_all = []
        
        for img_id in stim.get('stimID'):
            #open image and translate it to ndarray
            image = Image.open(pjoin(stim.meta['path'],img_id))
            image = np.asarray(image)
            
            #prepare a copy of image  
            image_min = self.image_backgroud(image, RGB)
            
            #get the single patch and compute activation for patch_all with pool method 'max'
            segments = self.decompose(image, meth='felzenszwalb')
            patch_all = self.generate_patch_single(image, segments, RGB)
            dnn_acts = self.dnn.compute_activation(patch_all, self.dmask).pool('max').get(self.layer)
            act_all = dnn_acts.flatten()
            
            #sort the activation from max to min
            act_sorted = np.argsort(-act_all)
            
            #put the patch back to an image and compute activation each time with pool method 'max'
            patch_add = self.generate_patch_mix(image, segments, act_sorted, RGB)
            dnn_acts_add = self.dnn.compute_activation(patch_add, self.dmask).pool('max').get(self.layer)    
            act_add = dnn_acts_add.flatten()
            
            #find the max index and the original index in act_sorted
            act_max_index = np.argmax(act_add)
            act_back_index = act_sorted[act_max_index] 
            
            #generate the minimal image
            for index in act_sorted:
                place_x,place_y = np.where(segments==index)
                for i in range(len(place_x)):
                    image_min[place_x[i],place_y[i]]=image[place_x[i],place_y[i]] 
                if index == act_back_index:
                    break
                
            image_min_all.append(image_min)
            
        return image_min_all
            
        
class MinmalComponentImage(Algorithm):
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """
    
    def set_params(self,  meth='pca', criterion='max'):
        """Set parameter for the estimator"""
        self.meth = meth
        self.criterion = criterion

    def compute(self, image):
        """Generate minmal image for image listed in stim object """
        pass