import torch
import torch.nn as nn
import numpy as np
from dnnbrain.dnn.core import Mask
from torch.autograd import Variable

class BrainDecoding:
    pass

class BrainDecodingIdentification(BrainDecoding):
    pass

class BrainDecodingClassification(BrainDecoding):
    pass


class BrainDecodingReconstruction:
    """
    Reconstruct stimulus pixel information based on brain activity
    """
    def __init__(self, generator, dnn, layer, de_type='PCA', components=20, 
                 function=nn.MSELoss(), n_epoch=20, optimizer=torch.optim.Adam(), lr=0.01,
                 lambda_pixel=0.1, lambda_feature=1, threshold=1):
        """
        Parameters
        ----------
        generator : torch model
            The generator model for synthesizing images
        dnn : DNN object
            DNN info to compute feature loss
        layer : list
            List containing interested layer names used in feature loss.
        function : nn.lossFunction
            Loss function used in torch. The default is nn.MSELoss(). 
        n_epoch : int
            The number of epochs. The default is 20.
        optimizer : object
            Optimizer function.

            If is None, use Adam to optimize all parameters in dnn. |br|
            If is not None, it must be torch optimizer object.
        lr : float
            Learning rate parameters in training. The default is 0.01.
        lambda_pixel : TYPE, optional
            DESCRIPTION. The default is 0.1.
        lambda_feature : TYPE, optional
            DESCRIPTION. The default is 1.
        threshold : TYPE, optional
            DESCRIPTION. The default is 1.
        """
        pass

    def set_model(self, generator, dnn, layer):
        pass
    
    def set_training(self, n_epoch, optimizer, lr):
        pass
    
    def set_loss(self, function, lambda_pixel, lambda_feature, threshold):
        pass
    
    def set_decompose(self, de_type, components):
        pass
    
    def decompose(self, brain_activ):
        """
        Decompose the brain acticity space into main component space.1

        Parameters
        ----------
        brain_activ : ndarray
            brain activation with shape as (n_vol, n_meas)
        
        Returns
        -------
        brain_de : ndarray
            Brain activation array after decomposion.
        """
        pass
 
    def compute_loss(self, image_generated, image_set):
        """
        Compute loss between generated image and validation set. |br|
        It contains two parts: pixel loss and feature loss
        
        Parameters
        ----------
        image_generated : ndarray
            Generated image with shape as (n_pic, n_chn, height, width)
        image_set : ndarray
            Stimulus image information with shape as (n_pic, n_chn, height, width)
            
        Returns
        -------
        loss : nn.loss
        """
        # define params
        dmask = Mask()
        #transpose dtype
        if image_generated.dtype is not np.uint8:
            image_generated = (image_generated*255).astype(np.uint8)
        # compute loss
        torch_com = lambda x: Variable(torch.from_numpy(x).double(), requires_grad=True)
        loss_pixel = self.function(torch_com(image_generated), torch_com(image_set))
        loss_feature = 0
        for layer in self.layer:
            dmask.set(layer, channels='all')
            act_train = self.dnn.compute_activation(image_generated, dmask).get(layer)
            act_val = self.dnn.compute_activation(image_set, dmask).get(layer)
            # mask the act if units meet the binarization threshold in the original stimulus
            act_train_masked = act_train[act_val > self.threshold]
            act_val_masked = act_val[act_val > self.threshold]
            loss_feature += self.function(torch_com(act_train_masked), torch_com(act_val_masked))
            dmask.clear()
            print(f'Finish computing activation of {layer} in loss')            
        loss_all = self.lambda_pixel*loss_pixel + self.lambda_feature*loss_feature
        return loss_all 
    
    
class BigGANDecodingReconstruction(BrainDecodingReconstruction):
    """
    Reconstruct stimulus pixel information using BigGAN generator
    """
    def __init__(self, generator, dnn, layer, category, de_type='PCA', components=20, 
                 function=nn.MSELoss(), n_epoch=20, optimizer=torch.optim.Adam(), lr=0.01,
                 lambda_pixel=0.1, lambda_feature=1, threshold=1):
        """
        Parameters
        ----------
        generator : torch model
            The generator model for synthesizing images
        dnn : DNN object
            DNN info to compute feature loss
        layer : list
            List containing interested layer names used in feature loss.
        category : int
            The category of the target decoding object.
            It should be ranging from 0 to 1000 based on ImageNet Challenge.
        function : nn.lossFunction
            Loss function used in torch. The default is nn.MSELoss(). 
        n_epoch : int
            The number of epochs. The default is 20.
        optimizer : object
            Optimizer function.

            If is None, use Adam to optimize all parameters in dnn. |br|
            If is not None, it must be torch optimizer object.
        lr : float
            Learning rate parameters in training. The default is 0.01.
        lambda_pixel : TYPE, optional
            DESCRIPTION. The default is 0.1.
        lambda_feature : TYPE, optional
            DESCRIPTION. The default is 1.
        threshold : TYPE, optional
            DESCRIPTION. The default is 1.
        """
        super(BigGANDecodingReconstruction, self).__init__(generator, dnn, layer, de_type, components, 
                                                           function, n_epoch, optimizer, lr,
                                                           lambda_pixel, lambda_feature, threshold)
        self.category = category
        self.set_model(generator, dnn, layer)
        self.set_training(n_epoch, optimizer, lr)
        self.set_loss(function, lambda_pixel, lambda_feature, threshold)
        self.set_decompose(de_type, components)
        
    def generate(self, latent_space):
        """
        Generate images from latent_space on specified generator type

        Parameters
        ----------
        latent_space : ndarray
            latent variable space with shape as (n_pic, n_latent)

        Returns
        -------
        image_generated : ndarray
            Generated image with shape as (n_pic, n_chn, height, width)
        """
        # define input
        class_vector = torch.zeros((1, 1000))
        class_vector[0, self.category] = 1
        truncation = 0.3
        pic_out = np.zeros((latent_space.shape[0], 3, 256,256))
        # start computing
        for idx in range(latent_space.shape[0]): 
            noise_vector = torch.Tensor(latent_space[idx])
            output = self.generator(noise_vector, class_vector, truncation)
            output = (output-output.min())/(output.max()-output.min())
            pic_single = output[0].data.numpy()
            pic_out[idx] = pic_single
            # print(f'Finish loading pics:{idx}/{latent_space.shape[0]} in epoch{epoch} ")
        return pic_out
    
    def fit(self, brain_activ, image_set):
        """
        Fit the decoding model 

        Parameters
        ----------
        brain_activ : ndarray
            brain activation with shape as (n_pic, n_voxel)
        image_set : ndarray
            Stimulus image information with shape as (n_pic, n_chn, height, width)

        Returns
        -------
        decoding_model : nn.model
            Decoding model 
        loss_all : list
            A list containing the loss information.
        """
        # Decompose brain activity
        brain_dec = torch.tensor(self.decompose(brain_activ)).float()  
        # Contruct decoding model
        decode_net = nn.Sequential(
                nn.Linear(self.components, 128)
        )
       
        optimizer = self.optimizer(decode_net.parameters(), self.lr)
        loss_all = []
        
        # Start training
        for epoch in range(self.n_epoch):
            latent_space = decode_net(brain_dec)
            latent_nor = nn.functional.normalize(latent_space, dim=1)
            image_gen = self.generate(latent_nor)
            loss= self.compute_loss(image_gen, image_set)
            print(f'loss:{loss.item()} in epoch{epoch}')
            loss_all.append(loss.item())
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return decode_net, loss_all
    
    def predict(self, decoding_model, brain_activ_test):
        """
        Test decoding model and generate image based on brain activity.

        Parameters
        ----------
        decoding_model : nn.model
            Decoding model 
        brain_activ_test : ndarray
            Test brain activation with shape as (n_pic, n_voxel)

        Returns
        -------
        img_gen_test : ndarray
            Generated images on test set.
        """
        brain_activ_test = torch.tensor(brain_activ_test).float() 
        # Start generating
        latent_space_test = decoding_model(brain_activ_test)
        latent_test_nor = nn.functional.normalize(latent_space_test, dim=1)
        img_gen_test = self.generate(latent_test_nor)
        return img_gen_test
    
    
class VAEDecodingReconstruction(BrainDecodingReconstruction):
    """
    Reconstruct stimulus pixel information using VAE generator
    """
    def __init__(self, generator, dnn, layer, de_type='PCA', components=20, 
                 function=nn.MSELoss(), n_epoch=20, optimizer=torch.optim.Adam(), lr=0.01,
                 lambda_pixel=0.1, lambda_feature=1, threshold=1):
        """
        Parameters
        ----------
        generator : torch model
            The generator model for synthesizing images
        dnn : DNN object
            DNN info to compute feature loss
        layer : list
            List containing interested layer names used in feature loss.
        function : nn.lossFunction
            Loss function used in torch. The default is nn.MSELoss(). 
        n_epoch : int
            The number of epochs. The default is 20.
        optimizer : object
            Optimizer function.

            If is None, use Adam to optimize all parameters in dnn. |br|
            If is not None, it must be torch optimizer object.
        lr : float
            Learning rate parameters in training. The default is 0.01.
        lambda_pixel : TYPE, optional
            DESCRIPTION. The default is 0.1.
        lambda_feature : TYPE, optional
            DESCRIPTION. The default is 1.
        threshold : TYPE, optional
            DESCRIPTION. The default is 1.
        """
        super(VAEDecodingReconstruction, self).__init__(generator, dnn, layer, de_type, components, 
                                                        function, n_epoch, optimizer, lr,
                                                        lambda_pixel, lambda_feature, threshold)
        self.set_model(generator, dnn, layer)
        self.set_training(n_epoch, optimizer, lr)
        self.set_loss(function, lambda_pixel, lambda_feature, threshold)
        self.set_decompose(de_type, components)
    
    def generate(self, latent_space):
        """
        Generate images from latent_space on specified generator type

        Parameters
        ----------
        latent_space : ndarray
            latent variable space with shape as (n_pic, n_latent)

        Returns
        -------
        image_generated : ndarray
            Generated image with shape as (n_pic, n_chn, height, width)
        """
        pass
    
    def fit(self, brain_activ, image_set):
        """
        Fit the decoding model 

        Parameters
        ----------
        brain_activ : ndarray
            brain activation with shape as (n_pic, n_voxel)
        image_set : ndarray
            Stimulus image information with shape as (n_pic, n_chn, height, width)

        Returns
        -------
        decoding_model : nn.model
            Decoding model 
        loss_all : list
            A list containing the loss information.
        """
    
    def predict(self, decoding_model, brain_activ_test):
        """
        Test decoding model and generate image based on brain activity.

        Parameters
        ----------
        decoding_model : nn.model
            Decoding model 
        brain_activ_test : ndarray
            Test brain activation with shape as (n_pic, n_voxel)

        Returns
        -------
        img_gen_test : ndarray
            Generated images on test set.
        """
        pass

class BrainEnhance:
    pass