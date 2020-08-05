

class BrainReconstruction:
    """
    Reconstruct stimulus pixel information based on brain activity
    """
    def __init__(self, brain_activ=None, generator_type=None, image_set=None):
        """
        Parameters
        ----------
        brain_activ : ndarray
            brain activation with shape as (n_vol, n_meas)
        generator_type : str
            Only 'BigGAN' and 'VAE' are supported
        image_set : ndarray
            Stimulus image information with shape as (n_pic, n_chn, height, width)
        """
        pass
    
    def set(self, brain_activ=None, generator_type=None, image_set=None):
        """
        Set some attributes
        
        Parameters
        ----------
        brain_activ : ndarray
            brain activation with shape as (n_vol, n_meas)
        generator_type : str
            Only 'BigGAN' and 'VAE' are supported
        image_set : ndarray
            Stimulus image information with shape as (n_pic, n_chn, height, width)
        """

        if brain_activ is not None:
            self.brain_activ = brain_activ

        if generator_type == 'BigGAN':
            pass
        elif generator_type == 'VAE': 
            pass
        else:
            raise ValueError('Only BigGAN and VAE generator are supported')
    
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
        
    def compute_loss(function, image_generated, dnn):
        """
        Compute loss between generated image and validation set. |br|
        It contains two parts: pixel loss and feature loss
        
        Parameters
        ----------
        function : nn.lossFunction
            Loss function used in torch 
        image_generated : ndarray
            Generated image with shape as (n_pic, n_chn, height, width)
        dnn : DNN object
            DNN info to compute feature loss
            
        Returns
        -------
        loss : nn.loss
        """
        pass
    
    def train(self):
        """
        Training the decoding model 

        Returns
        -------
        model 

        """
        pass
    
    def test(self, brain_actic_test):
        pass