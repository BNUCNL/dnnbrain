#import some packages needed
import numpy as np
import math
import torch
from PIL import Image

from os.path import join as pjoin

from torch.nn import ReLU
from torch.optim import Adam
from abc import ABC, abstractmethod
from skimage.segmentation import felzenszwalb, slic, quickshift

from dnnbrain.dnn.base import ImageSet
from dnnbrain.dnn.core import Stimulus, Mask

class ImageDecomposer(ABC):
    """ 
    An Abstract Base Classes class to define interface for image decomposer, 
    which decompose an image into different parcels or components
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def set_params(self):
        """ set params for the decomposer """
        
    @abstractmethod
    def decompose(self, image):
        """ decompose the image into multiple parcel or component"""
        
class ImageComponentDecomposer(ImageDecomposer):
    """ Use a component model to decompose an image into different components"""
    def __init__(self,component_model):

        self.model = component_model
    
    def set_params(self, ncomp):
        """Set necessary parameters for the decomposer"""
        self.ncomp = ncomp
    
    def decompose(self, image):
        print('Please write code here to decompose image.')
        
class ImageParcelDecomposer(ImageDecomposer):
    """ Use a parcel model to decompose an image into different components"""
    
    def __init__(self,parcel_model):
        """
        Parameter:

        ---------
        parcel_model[str] : the model to segmentate images 
        
        """
        self.model = parcel_model
        
    def set_params(self, nparcel):
        """Set necessary parameters for the decomposer"""
        self.nparcel = nparcel
        
    def decompose(self, image):
        
        if self.model == 'felzenszwalb':
            segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
        if self.model == 'slic':
            segments = slic(image, n_segments=250, compactness=10, sigma=1)
        if self.model == 'quickshift':
            segments = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
    
        return segments
    
    def image_backgroud(self, image, RGB):
        """
        Parameter:

        ---------
        image[ndarray] : (height,width,n_chn) shape
        RGB[tuple]: replace color
        
        """        
        image_copy=image.copy() 
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image_copy[i,j] = RGB
                
        return image_copy
    
    def generate_patch_single(self, image, segments, RGB):
        """
        Generate all patches of an image,the other area is replaced by image_backgroud
        
        Parameter:

        ---------
        image[ndarray] : (height,width,n_chn) shape
        segments[ndarray]: label of segmented image
        
        Return:

        ------

        patch_all[ndarray]: its shape is (n_segments, n_chn, height, width) 
        
        """         
        patch_all = []
        #find the location and replace other area with noise
        for label in np.unique(segments):
            image_RGB = self.image_backgroud(image, RGB)#生成噪声背景
            place_x,place_y = np.where(segments==label)  #通过标记找出每个patch的位置
            for i in range(len(place_x)):
                image_RGB[place_x[i],place_y[i]]=image[place_x[i],place_y[i]] #改变segmentation对应原始图片位置的RGB值
            
            #chage the shape(height,width,chn) to (chn,height,width) and add a newaxis
            patch = image_RGB.transpose((2,0,1)) 
            patch = patch[np.newaxis,:,:,:]
            
            #append the ndarray in a list
            patch_all = patch_all.append(patch)
        
        #translate list to ndarray
        patch_all = np.asarray(patch_all)
        
        return patch_all

    
    def generate_patch_mix(self, image, segments, act_sorted, RGB):
        """
        Generate all patches of an image,the other area is replaced by image_backgroud
        
        Parameter:

        ---------
        image[ndarray] : (height,width,n_chn) shape
        segments[ndarray]: label of segmented image 
        act_sorted[list] :index of activation from max to min in the original label
        
        Return:

        ------

        patch_all[ndarray]: its shape is (n_segments, n_chn, height, width) 
        
        """         
        
        image_RGB = self.image_backgroud(image, RGB)#生成噪声背景
        
        patch_add=[]
        #find the location and replace other area with noise
        for index in act_sorted:
            place_x,place_y = np.where(segments==index)  #通过标记找出每个patch的位置
            for i in range(len(place_x)):
                image_RGB[place_x[i], place_y[i]]=image[place_x[i], place_y[i]] #改变segmentation对应原始图片位置的RGB值
            
            #chage the shape(height,width,chn) to (chn,height,width) and add a newaxis
            patch = image_RGB.transpose((2, 0, 1)) 
            patch = patch[np.newaxis,:,:,:]
            
            #append the ndarray in a list
            patch_add.append(patch)
            
        #translate list to ndarray
        patch_add = np.asarray(patch_add)
        
        return patch_add



class MinmalImage():
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """
    def __init__(self, dnn, decomposer, criterion):
        """
        Parameter:

        ---------
        dnn[DNN]: netmodel
        decomposer[ImageDecomposer]: method to segment image
        criterion[str]: max
        """
        self.dnn = dnn
        self.criterion = criterion
        self.decomposer = decomposer
    
    def set_params(self,  decomposer, criterion):
        """Set parameter for the estimator"""
        self.decomposer = decomposer
        self.criterion = criterion


    def compute(self, stim, dmask, RGB=(255,255,255)):

        """
        Generate minimal image for image listed in stim object
        
        Parameter:

        ---------
        stim[Stimulus]: stimulus
        decomposer[ImageDecomposer]: method to segment image
        criterion[str]: max
        
        """
        #decompose = self.decomposer
        layer = dmask.layers[0]
        dataset = ImageSet(stim.meta['path'], stim.get('stimID'))
        
        image_min_all = []
        
        for img_id in dataset.img_ids:
            
            image = Image.open(pjoin(dataset.img_dir,img_id))
            image = np.asarray(image)
            image_min = self.decomposer.image_backgroud(image, RGB)
            
            segments = self.decomposer.decompose(image)
            
            patch_all = self.decomposer.generate_patch_single(image, segments, RGB)
            
            dnn_acts = self.dnn.compute_activation(patch_all, dmask)
            dnn_acts = dnn_acts.pool('max')
            dnn_acts = dnn_acts.get(layer)
        
            #put the dim from 4 to 1
            act_all = dnn_acts.flatten()
            
            #sort the activation from max to min
            act_sorted = np.argsort(-act_all)
            
            patch_add = self.decomposer.generate_patch_mix(image, segments, act_sorted)
            
            dnn_acts_add = self.dnn.compute_activation(patch_add, dmask)
            dnn_acts_add = dnn_acts_add.pool('max')
            dnn_acts_add = dnn_acts_add.get(layer)
            
            act_add = dnn_acts_add.flatten()
            
            #find the max index
            act_max_index = np.argmax(act_add)
    
            #find the original index in act_sorted
            act_back_index = act_sorted[act_max_index] #!!!!
            
            #generate the minimal image
            for index in act_sorted:
                place_x,place_y = np.where(segments==index)
                for p in range(len(place_x)):
                    image_min[place_x[p],place_y[p]]=image[place_x[p],place_y[p]] #改变segmentation对应原始图片位置的RGB值
                if index == act_back_index:
                    break
                
            image_min_all.append(image_min)
            
        return image_min_all
            
            
                