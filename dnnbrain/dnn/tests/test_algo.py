import os
import pytest
import numpy as np

from os.path import join as pjoin
from PIL import Image
from matplotlib import pyplot as plt
from dnnbrain.dnn import algo as d_algo
from dnnbrain.dnn.base import ip
from dnnbrain.dnn.models import AlexNet
from dnnbrain.utils.util import normalize



DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


class TestVanillaSaliencyImage:

    image = Image.open(pjoin(DNNBRAIN_TEST, 'image', 'images', 'n02108551_26574.JPEG'))

    def test_backprop(self):

        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()

        # prepare vanilla
        vanilla = d_algo.VanillaSaliencyImage(dnn)
        vanilla.set_layer('fc3', 276)

        # test backprop to conv1
        img_out = vanilla.backprop(self.image, 'conv1')
        assert img_out.shape == (3, *dnn.img_size)
        plt.figure()
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        # test backprop to conv1_relu
        img_out = vanilla.backprop(self.image, 'conv1_relu')
        assert img_out.shape == (64, 55, 55)
        plt.figure()
        plt.imshow(normalize(img_out)[0])

        plt.show()

    def test_backprop_smooth(self):

        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()

        # prepare vanilla
        vanilla = d_algo.VanillaSaliencyImage(dnn)
        vanilla.set_layer('fc3', 276)

        # test backprop to conv1
        img_out = vanilla.backprop_smooth(self.image, 2, to_layer='conv1')
        assert img_out.shape == (3, *dnn.img_size)
        plt.figure()
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        # test backprop to conv1_relu
        img_out = vanilla.backprop_smooth(self.image, 2, to_layer='conv1_relu')
        assert img_out.shape == (64, 55, 55)
        plt.figure()
        plt.imshow(normalize(img_out)[0])

        plt.show()


class TestGuidedSaliencyImage:

    image = Image.open(pjoin(DNNBRAIN_TEST, 'image', 'images', 'n02108551_26574.JPEG'))

    def test_backprop(self):

        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()

        # prepare guided
        guided = d_algo.GuidedSaliencyImage(dnn)

        # -test fc3-
        guided.set_layer('fc3', 276)
        # test backprop to conv1
        img_out = guided.backprop(self.image, 'conv1')
        assert img_out.shape == (3, *dnn.img_size)
        plt.figure()
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        # test backprop to conv1_relu
        img_out = guided.backprop(self.image, 'conv1_relu')
        assert img_out.shape == (64, 55, 55)
        plt.figure()
        plt.imshow(normalize(img_out)[0])

        # -test conv5-
        guided.set_layer('conv5', 1)
        img_out = guided.backprop(self.image, 'conv1')
        assert img_out.shape == (3, *dnn.img_size)
        plt.figure()
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        plt.show()

    def test_backprop_smooth(self):

        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()

        # prepare guided
        guided = d_algo.GuidedSaliencyImage(dnn)
        guided.set_layer('fc3', 276)

        # test backprop to conv1
        img_out = guided.backprop_smooth(self.image, 2, to_layer='conv1')
        assert img_out.shape == (3, *dnn.img_size)
        plt.figure()
        plt.imshow(normalize(img_out).transpose((1, 2, 0)))

        # test backprop to conv1_relu
        img_out = guided.backprop_smooth(self.image, 2, to_layer='conv1_relu')
        assert img_out.shape == (64, 55, 55)
        plt.figure()
        plt.imshow(normalize(img_out)[0])

        plt.show()


class TestSynthesisImage:

    
    def test_synthesize(self):

        dnn = AlexNet()
        
        syn_img = d_algo.SynthesisImage(dnn, 'fc3', 276)
        syn_img.set_metric(activ_metric='mean', regular_metric='TV',
                   precondition_metric='GB', smooth_metric='Fourier')
        syn_img.set_utiliz(save_out_interval=False, print_inter_loss=True)
        img_out = syn_img.synthesize(n_iter = 150, save_path = '.', save_interval =30, GB_radius = 1.2, step = 30)
        
        # assert
        assert img_out.shape == (3, *dnn.img_size)
        plt.figure()
        plt.imshow(img_out.transpose((1,2,0)))
        
        plt.show()
        
        
class TestMaskedImage:
    
    def test_set_parameters(self):
        
        dnn = AlexNet()
        
        int_img = np.random.rand(3,224,224)
        mask_img = d_algo.MaskedImage(dnn,'conv2',240)
        unit =(29,29)
        mask_img.set_parameters(initial_image=int_img,unit=unit)
        
        #assert 
        assert mask_img.initial_image.all() == int_img.all()
        assert mask_img.row == unit[0] and mask_img.column ==unit[1]
        
            
    
    def test_put_mask(self):
        
        dnn = AlexNet()
        unit =(14,14)
        syn_img = d_algo.SynthesisImage(dnn, 'conv2', 100)
        syn_img.set_metric(activ_metric='mean', regular_metric='TV',
                   precondition_metric='GB', smooth_metric='Fourier')
        syn_img.set_utiliz(save_out_interval=False, print_inter_loss=True)
        int_img = syn_img.synthesize(n_iter = 500,unit=unit,step=100)

        mask_img = d_algo.MaskedImage(dnn,'conv2',100)
        mask_img.set_parameters(initial_image=int_img,unit=unit)        
        img_out = mask_img.put_mask(maxiteration=200)
        
        #assert 2
        assert img_out.shape ==  (3, *dnn.img_size)
        plt.figure()
        plt.imshow(img_out.transpose((1,2,0)))
        
        plt.show()
  
      
class TestMinimalParcelImage:
        
    image = Image.open(pjoin(DNNBRAIN_TEST, 'image', 'images', 'n02108551_26574.JPEG'))
        
    def test_generate_minimal_image(self):
        
        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()
        
        #prepare m_img
        m_img = d_algo.MinimalParcelImage(dnn)
        m_img.set_layer('conv5', 125)
        
        #test minimal image of conv5
        m_img.felzenszwalb_decompose(np.asarray(self.image)) 
        img_out = m_img.generate_minimal_image()
        assert img_out.shape == (*self.image.size, 3) 
        
        #visualize image
        plt.figure()
        plt.imshow(img_out)   
        
        plt.show()
        

class TestOccluderDiscrepancyMapping:
    
    image = Image.open(pjoin(DNNBRAIN_TEST, 'image', 'images', 'n02108551_26574.JPEG'))
        
    def test_compute(self):
        
        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()
        
        #prepare occulder map
        oc_map = d_algo.OccluderDiscrepancyMapping(dnn)
        oc_map.set_layer('conv4', 27)
        
        #compute discrepancy map
        discrepancy_map = oc_map.compute(np.asarray(self.image))
        assert discrepancy_map.shape == (107,107) 
        
        #visualize image
        plt.figure()
        plt.imshow(discrepancy_map)   
        
        plt.show()

    
class TestUpsamplingActivationMapping:
    
    image = Image.open(pjoin(DNNBRAIN_TEST, 'image', 'images', 'n02108551_26574.JPEG'))
        
    def test_compute(self):
        
        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()
        
        #test upsampling mapping
        up_map = d_algo.UpsamplingActivationMapping(dnn)
        up_map.set_layer('conv3', 45)
        upsampling_map = up_map.compute(np.asarray(self.image))
        assert upsampling_map.shape == (dnn.img_size) 
        
        #visualize image
        plt.figure()
        plt.imshow(upsampling_map)   
        
        plt.show()
    
    
class TestEmpiricalReceptiveField:
    
    image = Image.open(pjoin(DNNBRAIN_TEST, 'image', 'images', 'n02108551_26574.JPEG'))
        
    def test_generate_rf(self):
        
        # prepare DNN
        dnn = AlexNet()

        # prepare image
        self.image.show()
        
        #prepare emp_rf
        emp_rf = d_algo.EmpiricalReceptiveField(dnn)
        emp_rf.set_layer('conv3', 45)
        emp_rf.set_params()
        
        #empirical receptive field size
        img = ip.resize(np.asarray(self.image).transpose(2,0,1), dnn.img_size)
        emp_rf_size = emp_rf.generate_rf(img)
        assert emp_rf_size > 99
  

class TestTheoreticalReceptiveField:
            
    def test_compute(self):
        
        # prepare DNN
        dnn = AlexNet()
        
        #test theoretical receptive field
        the_rf = d_algo.TheoreticalReceptiveField(dnn)
        the_rf.set_layer('conv3', 45)
        the_rf_size = the_rf.compute()
        assert the_rf_size == 99
        
        
if __name__ == '__main__':
    pytest.main()