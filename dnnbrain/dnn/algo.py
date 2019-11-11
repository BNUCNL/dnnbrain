#import some packages needed
import numpy as np
import math
import torch
from torch.nn import ReLU
from torch.optim import Adam
try:
    from misc_functions import preprocess_image, recreate_image
except ModuleNotFoundError:
    pass
    #raise Exception('Please install misc_functions in your work station')
    
class ImageDdecomposer():
    """ A class to decompose an image into different parts or components"""
    def __init__(self,):
        pass
    
class CNNMinmalImageEstimator():
    """
    A class to generate minmal image for a CNN model using a specific part 
    decomposer and optimization criterion
    """
    def __init__(self, model=None, part_decomposer=None, optimization_criterion = None):
        self.model = model
        self.decomposer = part_decomposer
        self.optimization_criterion = optimization_criterion;
        
    def set(self, model, part_decomposer, optimization_criterion):
        self.model = model
        self.decomposer = part_decomposer
        self.optimization_criterion = optimization_criterion
        
    def estimate(self, image, layer, channel):
        """Generate minmal image of input image for a target layer and channel """
        pass

    
    
class ReceptiveFieldEstimator():
    
    
    



###############################################################################
#define the class to find optimal stimuli
class CNNLayerVisualization():
    """
    Produces an image that minimizes the loss of a convolution
    operation for a specific layer and filter
        
    Functions:
    --------------
    hook_layer: hook the model when it runs
    visualise_layer_with_hooks: given a layer, channel/unit, find optimal stimuli for this channel/unit. 
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

    def visualise_layer_with_hooks(self, model_name, n_step, outdir, unit=-10):
        '''
        Find optimal stimuli for this unit
        
        Parameters:
        --------------
        n_step[int]: number of train step
        outdir[str]: put return into outdir
        unit[int]: the unit number
        
        Returens:
        --------------
        created_image_unit: the image of optimal stimuli for one unit
        '''
        
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        if model_name=='alexnet':
            random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        else:
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
            if unit==-10:
                loss = -torch.mean(self.conv_output) 
            else:
                raw = math.floor(unit/self.conv_output.shape[0])
                column = unit - (math.floor(unit/self.conv_output.shape[0]))*self.conv_output.shape[0]
                loss = -self.conv_output[raw][column]
                
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(-(loss.data.numpy())))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            

def rescale_grads(map,gradtype="all"):
    map = map - map.min()
    map /= map.max()
    return map


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, cnn_layer, channel):
        self.model.zero_grad()
        # Forward pass
        x = input_image
        for index, layer in enumerate(self.model.features):
            # Forward pass layer by layer
            # x is not used after this point because it is only needed to trigger
            # the forward hook function
            x = layer(x)
            # Only need to forward until the selected layer is reached
            if index == cnn_layer:
                # (forward hook function triggered)
                break
        conv_output = torch.sum(torch.abs(x[0, channel]))
        # Backward pass
        conv_output.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


def layer_channel_reconstruction(model, picimg, layer, channel):
    """
    DNN layer_channel reconstruction

    Parameters:
    net[str]: DNN network
    input[str]: input path
    layer[int]: layer number
    channel[int]: channel number
    out[str]: output path
    """
    picimg.requires_grad=True
    GBP = GuidedBackprop(model)
    guided_grads = GBP.generate_gradients(picimg, layer, channel)
    all_sal = rescale_grads(guided_grads,gradtype="all")
    out_image = torch.from_numpy(all_sal).permute(1,2,0)
    return  out_image
  

#visulize the receptive field
import copy
import sys
sys.path.append('/home/dell/Desktop/easy-receptive-fields-pytorch-master')
from Receptivefield import receptivefield


def RF_position(model, selected_layer, unit):  
    
    net = copy.copy(model)
    #register a hook
    def features_only(self, x):
        return self.features[0:selected_layer](x) 
    net.forward = features_only.__get__(net)
    
    #get RF
    rf = receptivefield(net, (1, 3, 224, 224))
    
    #unit number to unit_x and unit_y position
    created_data = torch.Tensor(1,3,224,224)
    layeract = net(created_data)  
    unit_x = math.floor(unit/layeract.shape[-1])
    unit_y = unit - (math.floor(unit/layeract.shape[-1]))*layeract.shape[-1]
    
    #get x_top, x_bottom, y_top, y_bottom which are 4 bounds of RF
    if rf.hcenter()[unit_x].item() - rf.rfsize.h/2 >= 0:
        x_top = rf.hcenter()[unit_x].item() - rf.rfsize.h/2
        x_top = int(x_top)
    else:
        x_top = 0
    
    if rf.hcenter()[unit_x].item() + rf.rfsize.h/2 <= 224:
        x_bottom = rf.hcenter()[unit_x].item() + rf.rfsize.h/2
        x_bottom = int(x_bottom)
    else:
        x_bottom = 224
        
    if rf.vcenter()[unit_y].item() - rf.rfsize.w/2 >= 0:
        y_top = rf.vcenter()[unit_y].item() - rf.rfsize.w/2
        y_top = int(y_top)
    else:
        y_top = 0
        
    if rf.vcenter()[unit_y].item() + rf.rfsize.w/2 <= 224:
        y_bottom = rf.vcenter()[unit_y].item() + rf.rfsize.w/2
        y_bottom = int(y_bottom)
    else:
        y_bottom = 224
    
    return x_top, x_bottom, y_top, y_bottom
  
