import torch
import copy
import numpy as np

from torch.optim import Adam
from torch.autograd import Variable
from dnnbrain.dnn.core import Algorithm


class SynthesisImage(Algorithm):
    """ 
    An Abstract Base Classes class to generate a synthetic image 
    that maximally activates a neuron
    """
    def __init__(self, dnn, layer, channel, 
                 activation_metric='mean', regularization_metric='L2', n_iter=30):
        """
        Parameter:
        ---------
        dnn[DNN]: dnnbrain's DNN object
        """        
        super(SynthesisImage,self).__init__(dnn, layer, channel)
        self.activation = None
        self.n_iter = n_iter

        # activation metric setting
        if activation_metric == 'max':
            self.activation_metric = self.max_activation
        elif activation_metric == 'mean':
            self.activation_metric = self.mean_activation
        else:
            raise AssertionError('Only max and mean metic is supported')
            
        # regularization metric setting 
        if regularization_metric == 'L1':
            self.regularization_metric = self.L1_norm
        elif regularization_metric == 'TV':
            self.regularization_metric = self.total_variation
        else:
            raise AssertionError('Only L2, Total variation is supported')       


    
        # loss setting
        self.activation_loss = [] 
        self.regularization_loss = []

        # Generate a random image
        self.image_size = (3,) + self.dnn.img_size
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        self.optimal_image = self.preprocess_image(random_image, False)
        
    def mean_activation(self):
        activ = -torch.mean(self.activation)
        self.activation_loss.append(activ)
        return activ
    
    def max_activation(self):
        activ = -torch.max(self.activation)
        self.activation_loss.append(activ)
        return activ
    
    def L2_norm(self):
        reg = np.abs((self.optimal_image[0]).detach().numpy()).sum()
        self.regularization_loss.append(reg) 

    def total_variation(self):
        pass 
    
    def gaussian_blur(self):
        pass
    
    def mean_image(self):
        pass
    
    def center_bias(self):
        pass


    def set_params(self, activation_metric, regularization_metric, n_iter=30):
        """
        Set the number of iteration

        Parameter:
        ---------
        n_iter[int]: the number of iteration
        """
        # activation metric setting
        if activation_metric == 'max':
            self.activation_metric = self.max_activation
        elif activation_metric == 'mean':
            self.activation_metric = self.mean_activation
        else:
            raise AssertionError('Only max and mean metic is supported')
            
        # regularization metric setting 
        if regularization_metric == 'L1':
            self.regularization_metric = self.L1_norm
        elif regularization_metric == 'TV':
            self.regularization_metric = self.total_variation
        else:
            raise AssertionError('Only L2, Total variation is supported')
            
        # time for iter
        self.n_iter = n_iter

            
    def register_hooks(self):
        """
        Define register hook and register them to specific layer and channel.
        """
        def forward_hook(module, feat_in, feat_out):
            self.activation = feat_out[0, self.channel]
            # print("feat_out[0, self.channel] = ",feat_out[0, self.channel])
        # register forward hook to the target layer
        module = self.dnn.layer2module(self.layer)
        module.register_forward_hook(forward_hook)

    def format_np_output(self,np_arr):
        """
            This is a (kind of) bandaid fix to streamline saving procedure.
            It converts all the outputs to the same format which is 3xWxH
            with using sucecssive if clauses.
        Args:
            im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
        """
        # Phase/Case 1: The np arr only has 2 dimensions
        # Result: Add a dimension at the beginning
        if len(np_arr.shape) == 2:
            np_arr = np.expand_dims(np_arr, axis=0)
        # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
        # Result: Repeat first channel and convert 1xWxH to 3xWxH
        if np_arr.shape[0] == 1:
            np_arr = np.repeat(np_arr, 3, axis=0)
        # Phase/Case 3: Np arr is of shape 3xWxH
        # Result: Convert it to WxHx3 in order to make it saveable by PIL
        if np_arr.shape[0] == 3:
            np_arr = np_arr.transpose(1, 2, 0)
        # Phase/Case 4: NP arr is normalized between 0-1
        # Result: Multiply with 255 and change type to make it saveable by PIL
        if np.max(np_arr) <= 1:
            np_arr = (np_arr * 255).astype(np.uint8)
        return np_arr

    def preprocess_image(self,pil_im, resize_im=True):
        """
            Processes image for CNNs

        Args:
            PIL_img (PIL_img): Image to process
            resize_im (bool): Resize to 224 or not
        returns:
            im_as_var (torch variable): Variable that contains processed float tensor
        """
        # mean and std list for channels (Imagenet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Resize image
        if resize_im:
            pil_im.thumbnail((512, 512))
        im_as_arr = np.float32(pil_im)
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
            im_as_arr[channel] -= mean[channel]
            im_as_arr[channel] /= std[channel]
        # Convert to float tensor
        im_as_ten = torch.from_numpy(im_as_arr).float()
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)
        im_as_var = Variable(im_as_ten, requires_grad=True)
        return im_as_var

    def recreate_image(self,im_as_var):
        """
            Recreates images from a torch variable, sort of reverse preprocessing
        Args:
            im_as_var (torch variable): Image to recreate
        returns:
            recreated_im (numpy arr): Recreated image in array
        """
        reverse_mean = [-0.485, -0.456, -0.406]
        reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
        recreated_im = copy.copy(im_as_var.data.numpy()[0])
        for c in range(3):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
        recreated_im[recreated_im > 1] = 1
        recreated_im[recreated_im < 0] = 0
        recreated_im = np.round(recreated_im * 255)

        recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
        return recreated_im
        
    def L1synthesize(self):
        """
        Synthesize the image which maximally activates target layer and channel
        using L1 regularization.
        """
        
        # Hook the selected layer
        self.register_hooks()

     

        # optimal_image = torch.randn(1, *self.image_size)
        # optimal_image.requires_grad_(True)
        # optimal_image = Variable(optimal_image, requires_grad=True)  #


        # Define optimizer for the image
        optimizer = Adam([self.optimal_image], lr=0.1, betas=(0.9,0.99))
        for i in range(1, self.n_iter+1):
            # clear gradients for next train
            optimizer.zero_grad()
            # Forward pass layer by layer until the target layer
            # to triger the hook funciton.

            self.dnn.model(self.optimal_image)
            alpha = 0.1
            # Loss function is the mean of the output of the selected filter
            # We try to maximize the mean of the output of that specific filter
            loss =  self.activation_metric() + alpha * self.regularization_metric()

        
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.optimal_image = self.recreate_image(self.optimal_image)
        # Return the optimized image
        return np.uint8(self.optimal_image[0].detach().numpy())









