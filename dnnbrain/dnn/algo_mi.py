import numpy as np
from PIL import Image
from abc import ABC, abstractmethod


from skimage import segmentation 

    
    def combine_parcel(self, index):
    """combine the indexed parcel into a image"""
        pass 
    
    
    
    def generate_minmal_image(self)):

        return patch_add
           
        
    def compute(self, stim, RGB=(255,255,255)):

        
        # workflow
        
        # sort the image
        # iterater combine image to get activation
        # return the opimized curve and minmal image
        
        
        
        #generate the minimal image
        for index in act_sorted:
            place_x,place_y = np.where(segments==index)
            for p in range(len(place_x)):
                image_min[place_x[p],place_y[p]]=image[place_x[p],place_y[p]] #¸Ä±äsegmentation¶ÔÓ¦Ô­Ê¼Í¼Æ¬Î»ÖÃµÄRGBÖµ
            if index == act_back_index:
                break
