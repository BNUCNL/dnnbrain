#! /usr/bin/env python

"""
Find the optimal stimuli on one unit or layer
Author: Yiyuan Zhang @ BNU

A example of Linux code to use this module like this:
python /home/dell/Desktop/dnnoptimage.py -net alexnet -layer 1 -channel 5 -unit_x 5 -unit_y 7 -unit_outdir /home/dell/Desktop/DNN_code/unitimage.jpg -layer_outdir /home/dell/Desktop/DNN_code/layerimage.jpg
"""


#import some packages needed
import numpy as np
import torch
from torch.optim import Adam
from torchvision import models
from dnnbrain.bin.dnnoptimage import main

try:
    from misc_functions import preprocess_image, recreate_image, save_image
except ModuleNotFoundError:
    raise Exception('Please install misc_functions in your work station')


#load the model (alexnet, vgg16, vgg19)
def model():
    
    if main.args.net == 'alexnet':
        model = models.alexnet(pretrained=True)
        return model
    
    elif main.args.net == 'vgg16':
        model = models.vgg16(pretrained=True)
        return model
    
    elif main.args.net == 'vgg19':
        model = models.vgg19(pretrained=True)
        return model
    
    else:
        raise Exception('Network was not supported, please contact author for implementation.')
        
    for param in model.parameters():
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
        raw[int]: the unit position in x axis
        column[int]: the unit position in y axis
        n_step[int]: number of train step
        outdir[str]: put return into outdir
        
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
            loss = -self.conv_output[raw][column]
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image_unit = recreate_image(processed_image)
            
            save_image(self.created_image_unit, outdir)


    def visualise_layer_channel_with_hooks(self, n_step, outdir):
        '''
        Find optimal stimuli for this channel of layer
        
        Parameters:
        --------------
        n_step[int]: number of train step
        outdir[str]: put return into outdir
        
        Returens:
        --------------
        created_image_layer: the image of optimal stimuli for channel of layer
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
            
            save_image(self.created_image_layer, outdir)



#execute the codes above
model = model()

n_step = 41
layer_vis = CNNLayerVisualization(model.features, main.args.layer, main.args.channel)
    
layer_vis.visualise_layer_channel_with_hooks(n_step, main.args.layer_outdir)
layer_vis.visualise_layer_channel_unit_with_hooks(main.args.unit_x, main.args.unit_y, n_step, main.args.unit_outdir)
  
