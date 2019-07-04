#! /usr/bin/env python

"""
find the characterize relative to one unit
Author: Yiyuan Zhang @ BNU
Reviewer:
"""

#import some packages needed
import os
import sys
import argparse
import numpy as np


import torch
from torch.optim import Adam
from torchvision import models

sys.path.append('/home/dell/Desktop/src')
from misc_functions import preprocess_image, recreate_image, save_image

model_alexnet = models.alexnet(pretrained=True)
for param in model_alexnet.parameters():
    param.requires_grad = False
    
    

#define the class to find optimal stimuli
class CNNLayerVisualization():
    """
    Produces an image that minimizes the loss of a convolution
    operation for a specific layer and filter
        
    Functions:
    --------------
    hook_layer: hook the model when it runs
    visualise_layer_channel_unit_with_hooks: given a layer, channel and unit, find optimal stimuli for this unit 
    visualise_layer_channel_with_hooks: given a layer and channel, find optimal stimuli for this channel 
    """
    
    
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')


    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)


    def visualise_layer_channel_unit_with_hooks(self, raw, column, n_step, outdir):
        '''
        Find optimal stimuli for this unit
        
        Parameters:
        --------------
        raw: the unit's x axis
        column: the unit's y axis
        n_step: number of train step
        outdir: put return into it
        
        Returens:
        --------------
        the image of optimal stimuli for one unit
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
            loss = -self.conv_output[raw][column]
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image_unit = recreate_image(processed_image)
            
            #save_image(self.created_image_unit, outdir)


    def visualise_layer_channel_with_hooks(self, n_step, outdir):
        '''
        Find optimal stimuli for this channel of layer
        
        Parameters:
        --------------
        n_step: number of train step
        outdir: put return into it
        
        Returens:
        --------------
        the image of optimal stimuli for channel of layer
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
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image_layer = recreate_image(processed_image)
            
            #save_image(self.created_image_layer, outdir)
    
#define the function
def main():
    '''
    Main code 
    
    Returns:
    -------------
    the image of optimal stimuli for channel of layer and the image of optimal stimuli for one unit
    '''
    
    parser = argparse.ArgumentParser(description='DNN units optimal stimuli')
    
    parser.add_argument('-net',
                        type = str,
                        required = True,
                        choices=['alexnet', 'vgg16', 'vgg19', 'resnet'],
                        help = 'specific name of CNN')

    parser.add_argument('-layer',
                        type = int,
                        required = True,
                        help = 'activation for specific layers')
    
    parser.add_argument('-channel',
                        type = int,
                        required = True,
                        help = 'activation for specific channels')

    parser.add_argument('-unit_x',
                        type = int,
                        required = True,
                        help = 'activation for specific units x_axis')
    
    parser.add_argument('-unit_y',
                        type = int,
                        required = True,
                        help = 'activation for specific units y_axis')
    
    parser.add_argument('-outdir',
                        type = str,
                        required = True,
                        help = 'output directory in Linux')
    
    args = parser.parse_args()
    
    layer_vis = CNNLayerVisualization(model_alexnet.features, args.layer, args.channel)
    
    return layer_vis.visualise_layer_channel_with_hooks(n_step, args.outdir), layer_vis.visualise_layer_channel_unit_with_hooks(args.unit_x, args.unit_y, n_step, args.outdir)



    
if __name__ == '__main__':
    model_alexnet = models.alexnet(pretrained=True)
    for param in model_alexnet.parameters():
        param.requires_grad = False
        
    n_step = 31
    main()
    
