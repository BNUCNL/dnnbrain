import os
import unittest
import numpy as np

from os.path import join as pjoin
from dnnbrain.dnn import analyzer
from dnnbrain.dnn.io import NetLoader, ImgDataset
from dnnbrain.utils.io import read_stim_csv
from torch.utils.data import DataLoader
from torchvision import transforms

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.path.expanduser('~'), '.dnnbrain_tmp')


class TestAct(unittest.TestCase):

    def test_dnn_activation(self):

        # loader net
        net_loader = NetLoader('alexnet')
        transform = transforms.Compose([transforms.Resize(net_loader.img_size),
                                        transforms.ToTensor()])

        # read stimuli
        stim_file = pjoin(DNNBRAIN_TEST, 'image', 'sub-CSI1_ses-01_imagenet.stim.csv')
        stim_dict = read_stim_csv(stim_file)
        dataset = ImgDataset(stim_dict['path'], stim_dict['stim']['stimID'][:5],
                             transform=transform)
        data_loader = DataLoader(dataset, batch_size=5, shuffle=False)

        # extract activation
        acts_conv5 = []
        acts_conv5_chn = []
        acts_fc3 = []
        for stims, _ in data_loader:
            acts_conv5.extend(analyzer.dnn_activation(stims, net_loader.model,
                                                      net_loader.layer2loc['conv5']))
            acts_conv5_chn.extend(analyzer.dnn_activation(stims, net_loader.model,
                                                          net_loader.layer2loc['conv5'], [1, 2, 5]))
            acts_fc3.extend(analyzer.dnn_activation(stims, net_loader.model,
                                                    net_loader.layer2loc['fc3']))
        acts_conv5 = np.array(acts_conv5)
        acts_conv5_chn = np.array(acts_conv5_chn)
        acts_fc3 = np.array(acts_fc3)

        # assert
        self.assertEqual(acts_conv5.shape, (5, 256, 13, 13))
        self.assertEqual(acts_conv5_chn.shape, (5, 3, 13, 13))
        self.assertTrue(np.all(acts_conv5[:, [1, 2, 5]] == acts_conv5_chn))
        self.assertEqual(acts_fc3.shape, (5, 1000))


if __name__ == '__main__':
    unittest.main()
