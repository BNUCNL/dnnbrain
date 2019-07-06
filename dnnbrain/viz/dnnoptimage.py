#! /usr/bin/env python

"""
Find the optimal stimuli on one unit or layer
Author: Yiyuan Zhang @ BNU
"""

#import some packages needed
import numpy as np
import torch
from torch.optim import Adam

try:
    from misc_functions import preprocess_image, recreate_image, save_image
except ModuleNotFoundError:
    raise Exception('Please install misc_functions in your work station')

    
    
#define the class to find optimal stimuli
class CNNLayerVisualization():
    """
    Produces an image that minimizes the loss of a convolution
    operation for a specific layer and filter
        
    Functions:
    --------------
    hook_layer: hook the model when it runs
    visualise_layer_with_hooks: given a layer, channel (and unit), find optimal stimuli for this channel (unit). 
    """
    
    
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0

    
    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)


    def visualise_layer_with_hooks(self, n_step, outdir, unit_x=-100, unit_y =-100):
        '''
        Find optimal stimuli for this unit
        
        Parameters:
        --------------
        n_step[int]: number of train step
        outdir[str]: put return into outdir
        unit_x[int]: the unit position in x axis
        unit_y[int]: the unit position in y axis
        
        Returens:
        --------------
        created_image_unit: the image of optimal stimuli for one unit
        '''
        
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        
        for i in range(1, n_step):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            if unit_x==-100:
                loss = -torch.mean(self.conv_output) 
            else:
                loss = -self.conv_output[unit_x][unit_y]
                
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image_unit = recreate_image(processed_image)
            
        save_image(self.created_image_unit, outdir)
