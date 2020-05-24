import os
import time
import torch
import numpy as np

from os.path import join as pjoin
from scipy.stats import pearsonr
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models as tv_models
from dnnbrain.dnn.core import Stimulus, Activation
from dnnbrain.dnn.base import ImageSet, VideoSet, dnn_mask, array_statistic

DNNBRAIN_MODEL = pjoin(os.environ['DNNBRAIN_DATA'], 'models')


class VggFaceModel(nn.Module):
    """Vgg_face's model architecture"""

    def __init__(self):
        super(VggFaceModel, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [3, 224, 224]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc8(x)
        return x


class DNN:
    """
    Deep neural network

    Attributes:
    ----------
    model[nn.Modules]: DNN model
    layer2loc[dict]: map layer name to its location in the DNN model
    img_size[tuple]: the input image size
    train_transform[torchvision.transform]:
        the transform used in training state
    test_transform[torchvision.transform]:
        the transform used in testing state
    """

    def __init__(self):

        self.model = None
        self.layer2loc = None
        self.img_size = None  # (height, width)
        self.train_transform = None
        self.test_transform = None

    @property
    def layers(self):
        raise NotImplementedError('This method should be implemented in subclasses.')

    def save(self, fname):
        """
        Save DNN parameters

        Parameter:
        ---------
        fname[str]: output file name with suffix as .pth
        """
        assert fname.endswith('.pth'), 'File suffix must be .pth'
        torch.save(self.model.state_dict(), fname)

    def eval(self):
        """
        Turn to evaluation mode

        Return:
        ------
        self[DNN]
        """
        self.model.eval()

        return self

    def layer2module(self, layer):
        """
        Get a PyTorch Module object according to the layer name.

        Parameter:
        ---------
        layer[str]: layer name

        Return:
        ------
        module[Module]: PyTorch Module object
        """
        raise NotImplementedError('This method should be implemented in subclasses.')

    def compute_activation(self, stimuli, dmask, pool_method=None, cuda=False):
        """
        Extract DNN activation

        Parameters:
        ----------
        stimuli[Stimulus|ndarray]: input stimuli
            If is Stimulus, loaded from files on the disk.
            If is ndarray, its shape is (n_stim, n_chn, height, width)
        dmask[Mask]: The mask includes layers/channels/rows/columns of interest.
        pool_method[str]: pooling method, choices=(max, mean, median, L1, L2)
        cuda[bool]: use GPU or not

        Return:
        ------
        activation[Activation]: DNN activation
        """
        # prepare stimuli loader
        if isinstance(stimuli, np.ndarray):
            stim_set = []
            for arr in stimuli:
                img = Image.fromarray(arr.transpose((1, 2, 0)))
                stim_set.append((self.test_transform(img), 0))
        elif isinstance(stimuli, Stimulus):
            if stimuli.header['type'] == 'image':
                stim_set = ImageSet(stimuli.header['path'], stimuli.get('stimID'),
                                    transform=self.test_transform)
            elif stimuli.header['type'] == 'video':
                stim_set = VideoSet(stimuli.header['path'], stimuli.get('stimID'),
                                    transform=self.test_transform)
            else:
                raise TypeError('{} is not a supported stimulus type.'.format(stimuli.header['type']))
        else:
            raise TypeError('The input stimuli must be an instance of ndarray or Stimulus!')
        data_loader = DataLoader(stim_set, 8, shuffle=False)

        # -extract activation-
        # prepare model
        self.model.eval()
        if cuda:
            assert torch.cuda.is_available(), 'There is no CUDA available.'
            self.model.to(torch.device('cuda'))

        n_stim = len(stim_set)
        activation = Activation()
        for layer in dmask.layers:
            # prepare dnn activation hook
            acts_holder = []

            def hook_act(module, input, output):

                # copy activation
                if cuda:
                    acts = output.cpu().data.numpy().copy()
                else:
                    acts = output.detach().numpy().copy()

                # unify dimension number
                if acts.ndim == 4:
                    pass
                elif acts.ndim == 2:
                    acts = acts[:, :, None, None]
                else:
                    raise ValueError('Unexpected activation shape:', acts.shape)

                # mask activation
                mask = dmask.get(layer)
                acts = dnn_mask(acts, mask.get('chn'),
                                mask.get('row'), mask.get('col'))

                # pool activation
                if pool_method is not None:
                    acts = array_statistic(acts, pool_method, (2, 3), True)

                # hold activation
                acts_holder.extend(acts)

            module = self.layer2module(layer)
            hook_handle = module.register_forward_hook(hook_act)

            # extract DNN activation
            for stims, _ in data_loader:
                # stimuli with shape as (n_stim, n_chn, height, width)
                if cuda:
                    stims = stims.to(torch.device('cuda'))
                self.model(stims)
                print('Extracted activation of {0}: {1}/{2}'.format(
                    layer, len(acts_holder), n_stim))
            activation.set(layer, np.asarray(acts_holder))

            hook_handle.remove()

        return activation

    def get_kernel(self, layer, kernels=None):
        """
        Get kernels' weights of the layer

        Parameters:
        ----------
        layer[str]: layer name
        kernels[int|list]: serial numbers of kernels
            start from 1

        Return:
        ------
        weights[tensor]: kernel weights
        """
        # get the module
        module = self.layer2module(layer)

        # get the weights
        weights = module.weight
        if kernels is not None:
            # deal with kernel numbers
            kernels = np.asarray(kernels)
            assert np.all(kernels > 0), 'The kernel number should start from 1.'
            kernels = kernels - 1
            # get part of weights
            weights = weights[kernels]

        return weights

    def ablate(self, layer, channels=None):
        """
        Ablate DNN kernels' weights

        Parameters:
        ----------
        layer[str]: layer name
        channels[list]: sequence numbers of channels of interest
            If None, ablate the whole layer.
        """
        # localize the module
        module = self.layer2module(layer)

        # ablate kernels' weights
        if channels is None:
            module.weight.data[:] = 0
        else:
            channels = [chn - 1 for chn in channels]
            module.weight.data[channels] = 0

    def train(self, data, n_epoch, task, optimizer=None, method='tradition', target=None,
              data_train=False, data_validation=None):
        """
        Train the DNN model

        Parameters:
        ----------
        data[Stimulus|ndarray]: training data
            If is Stimulus, load stimuli from files on the disk.
                Note, the data of the 'label' item in the Stimulus object will be used as
                truth of the output when 'target' is None.
            If is ndarray, it contains stimuli with shape as (n_stim, n_chn, height, width).
                Note, the truth data must be specified by 'target' parameter.
        n_epoch[int]: the number of epochs
        task[str]: task function
            choices=('classification', 'regression').
        optimizer[object]: optimizer function
            If is None, use Adam to optimize all parameters in dnn.
            If is not None, it must be torch optimizer object.
        method[str]: training method, by default is 'tradition'.
            For some specific models (e.g. inception), loss needs to be calculated in another way.
        target[ndarray]: the output of the model
            Its shape is (n_stim,) for classification or (n_stim, n_feat) for regression.
                Note, n_feat is the number of features of the last layer.
        data_train[bool]:
            If true, test model performance on the training data.
        data_validation[Stimulus|ndarray]: validation data
            If is not None, test model performance on the validation data.

        Return:
        ------
        train_dict[dict]:
            epoch_loss[list]: losses of epochs
            step_loss[list]: step losses of epochs
                The indices are one-to-one corresponding with the epoch_losses.
                Each element is a list where elements are step losses of the corresponding epoch.
            score_train[list]: scores of epochs on training data
            score_validation[list]: scores of epochs on validation data
        """
        # prepare data loader
        if isinstance(data, np.ndarray):
            stim_set = [Image.fromarray(arr.transpose((1, 2, 0))) for arr in data]
            stim_set = [(self.train_transform(img), trg) for img, trg in zip(stim_set, target)]
        elif isinstance(data, Stimulus):
            if data.header['type'] == 'image':
                stim_set = ImageSet(data.header['path'], data.get('stimID'),
                                    data.get('label'), transform=self.train_transform)
            elif data.header['type'] == 'video':
                stim_set = VideoSet(data.header['path'], data.get('stimID'),
                                    data.get('label'), transform=self.train_transform)
            else:
                raise TypeError(f"{data.header['type']} is not a supported stimulus type.")

            if target is not None:
                # We presume small quantity stimuli will be used in this way.
                # Usually hundreds or thousands such as fMRI stimuli.
                stim_set = [(img, trg) for img, trg in zip(stim_set[:][0], target)]
        else:
            raise TypeError('The input data must be an instance of ndarray or Stimulus!')
        data_loader = DataLoader(stim_set, 8, shuffle=True)

        # prepare criterion
        if task == 'classification':
            criterion = nn.CrossEntropyLoss()
        elif task == 'regression':
            criterion = nn.MSELoss()
        else:
            raise ValueError(f'Unsupported task: {task}')

        # prepare optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters())

        # start train
        train_dict = dict()
        train_dict['step_loss'] = []
        train_dict['epoch_loss'] = []
        train_dict['score_train'] = []
        train_dict['score_validation'] = []
        time1 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train()
        self.model.to(device)
        for epoch in range(n_epoch):
            print(f'Epoch-{epoch+1}/{n_epoch}')
            print('-' * 10)
            time2 = time.time()
            running_loss = 0.0

            step_losses = []
            for inputs, targets in data_loader:
                inputs.requires_grad_(True)
                inputs = inputs.to(device)
                targets = targets.to(device)
                with torch.set_grad_enabled(True):
                    if method == 'tradition':
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                    elif method == 'inception':
                        # Google inception model
                        outputs, aux_outputs = self.model(inputs)
                        loss1 = criterion(outputs, targets)
                        loss2 = criterion(aux_outputs, targets)
                        loss = loss1 + 0.4 * loss2
                    else:
                        raise Exception(f'not supported method-{method}')

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Statistics loss in every batch
                step_losses.append(loss.item())
                running_loss += loss.item() * inputs.size(0)

            # calculate loss
            train_dict['step_loss'].append(step_losses)
            epoch_loss = running_loss / len(data_loader.dataset)
            print(f'Loss: {epoch_loss}')
            train_dict['epoch_loss'].append(epoch_loss)

            # test performance
            if data_train:
                test_dict = self.test(data, task, target, torch.cuda.is_available())
                print(f"Score_on_train: {test_dict['score']}")
                train_dict['score_train'].append(test_dict['score'])
                self.model.train()
            if data_validation is not None:
                test_dict = self.test(data_validation, task, target, torch.cuda.is_available())
                print(f"Score_on_test: {test_dict['score']}")
                train_dict['score_validation'].append(test_dict['score'])
                self.model.train()

            # print time of a epoch
            epoch_time = time.time() - time2
            print('This epoch costs {:.0f}m {:.0f}s\n'.format(epoch_time // 60, epoch_time % 60))

        time_elapsed = time.time() - time1
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # back to cpu
        self.model.to(torch.device('cpu'))
        return train_dict

    def test(self, data, task, target=None, cuda=False):
        """
        Test the DNN model

        Parameters:
        ----------
        data[Stimulus|ndarray]: testing data
            If is Stimulus, load stimuli from files on the disk.
                Note, the data of the 'label' item in the Stimulus object will be used as
                output of the model when 'target' is None.
            If is ndarray, it contains stimuli with shape as (n_stim, n_chn, height, width).
                Note, the output data must be specified by 'target' parameter.
        task[str]: choices=(classification, regression)
        target[ndarray]: the output of the model
            Its shape is (n_stim,) for classification or (n_stim, n_feat) for regression.
                Note, n_feat is the number of features of the last layer.
        cuda[bool]: use GPU or not

        Returns:
        -------
        test_dict[dict]:
            if task == 'classification':
                pred_value[array]: prediction labels by the model
                    2d array with shape as (n_stim, n_class)
                    Each row's labels are sorted from large to small their probabilities.
                true_value[array]: observation labels
                score[float]: prediction accuracy
            if task == 'regression':
                pred_value[array]: prediction values by the model
                true_value[array]: observation values
                score[float]: R square between pred_values and true_values
        """
        # prepare data loader
        if isinstance(data, np.ndarray):
            stim_set = [Image.fromarray(arr.transpose((1, 2, 0))) for arr in data]
            stim_set = [(self.test_transform(img), trg) for img, trg in zip(stim_set, target)]
        elif isinstance(data, Stimulus):
            if data.header['type'] == 'image':
                stim_set = ImageSet(data.header['path'], data.get('stimID'),
                                    data.get('label'), transform=self.test_transform)
            elif data.header['type'] == 'video':
                stim_set = VideoSet(data.header['path'], data.get('stimID'),
                                    data.get('label'), transform=self.test_transform)
            else:
                raise TypeError(f"{data.header['type']} is not a supported stimulus type.")

            if target is not None:
                # We presume small quantity stimuli will be used in this way.
                # Usually hundreds or thousands such as fMRI stimuli.
                stim_set = [(img, trg) for img, trg in zip(stim_set[:][0], target)]
        else:
            raise TypeError('The input data must be an instance of ndarray or Stimulus!')
        data_loader = DataLoader(stim_set, 8, shuffle=False)

        # start test
        self.model.eval()
        if cuda:
            assert torch.cuda.is_available(), 'There is no CUDA available.'
            self.model.to(torch.device('cuda'))
        pred_values = []
        true_values = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(data_loader):
                if cuda:
                    inputs = inputs.to(torch.device('cuda'))
                outputs = self.model(inputs)

                # collect outputs
                if cuda:
                    pred_values.extend(outputs.cpu().detach().numpy())
                else:
                    pred_values.extend(outputs.detach().numpy())
                true_values.extend(targets.numpy())
        pred_values = np.array(pred_values)
        true_values = np.array(true_values)

        # prepare output info
        test_dict = dict()
        if task == 'classification':
            pred_values = np.argsort(-pred_values, 1)
            # calculate accuracy
            score = np.mean(pred_values[:, 0] == true_values)
        elif task == 'regression':
            # calculate r_square
            r, _ = pearsonr(pred_values.ravel(), true_values.ravel())
            score = r ** 2
        else:
            raise ValueError(f'Not supported task: {task}')
        test_dict['pred_value'] = pred_values
        test_dict['true_value'] = true_values
        test_dict['score'] = score

        return test_dict

    def __call__(self, inputs):
        """
        Feed the model with the inputs

        Parameter:
        ---------
        inputs[Tensor]: a tensor with shape as (n_stim, n_chn, n_height, n_width)

        Return:
        ------
        outputs[Tensor]: output of the model, usually with shape as (n_stim, n_feat)
            n_feat is the number of out features in the last layer of the model.
        """
        outputs = self.model(inputs)

        return outputs


class AlexNet(DNN):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = tv_models.alexnet()
        self.model.load_state_dict(torch.load(
            pjoin(DNNBRAIN_MODEL, 'alexnet.pth')))
        self.layer2loc = {'conv1': ('features', '0'), 'conv1_relu': ('features', '1'),
                          'conv1_maxpool': ('features', '2'), 'conv2': ('features', '3'),
                          'conv2_relu': ('features', '4'), 'conv2_maxpool': ('features', '5'),
                          'conv3': ('features', '6'), 'conv3_relu': ('features', '7'),
                          'conv4': ('features', '8'), 'conv4_relu': ('features', '9'),
                          'conv5': ('features', '10'), 'conv5_relu': ('features', '11'),
                          'conv5_maxpool': ('features', '12'), 'fc1': ('classifier', '1'),
                          'fc1_relu': ('classifier', '2'), 'fc2': ('classifier', '4'),
                          'fc2_relu': ('classifier', '5'), 'fc3': ('classifier', '6')}
        self.img_size = (224, 224)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            normalize
        ])

    @property
    def layers(self):
        return list(self.layer2loc.keys())

    def layer2module(self, layer):
        """
        Get a PyTorch Module object according to the layer name.

        Parameter:
        ---------
        layer[str]: layer name

        Return:
        ------
        module[Module]: PyTorch Module object
        """
        module = self.model
        for k in self.layer2loc[layer]:
            module = module._modules[k]

        return module


class VggFace(DNN):

    def __init__(self):
        super(VggFace, self).__init__()

        self.model = VggFaceModel()
        self.model.load_state_dict(torch.load(
            pjoin(DNNBRAIN_MODEL, 'vgg_face_dag.pth')))
        self.layer2loc = {'conv1_1': ('conv1_1',), 'relu1_1': ('relu1_1',),
                          'conv1_2': ('conv1_2',), 'relu1_2': ('relu1_2',),
                          'pool1': ('pool1',), 'conv2_1': ('conv2_1',),
                          'relu2_1': ('relu2_1',), 'conv2_2': ('conv2_2',),
                          'relu2_2': ('relu2_2',), 'pool2': ('pool2',),
                          'conv3_1': ('conv3_1',), 'relu3_1': ('relu3_1',),
                          'conv3_2': ('conv3_2',), 'relu3_2': ('relu3_2',),
                          'conv3_3': ('conv3_3',), 'relu3_3': ('relu3_3',),
                          'pool3': ('pool3',), 'conv4_1': ('conv4_1',),
                          'relu4_1': ('relu4_1',), 'conv4_2': ('conv4_2',),
                          'relu4_2': ('relu4_2',), 'conv4_3': ('conv4_3',),
                          'relu4_3': ('relu4_3',), 'pool4': ('pool4',),
                          'conv5_1': ('conv5_1',), 'relu5_1': ('relu5_1',),
                          'conv5_2': ('conv5_2',), 'relu5_2': ('relu5_2',),
                          'conv5_3': ('conv5_3',), 'relu5_3': ('relu5_3',),
                          'pool5': ('pool5',), 'fc6': ('fc6',),
                          'relu6': ('relu6',), 'fc7': ('fc7',),
                          'relu7': ('relu7',), 'fc8': ('fc8',)}
        self.img_size = (224, 224)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

    @property
    def layers(self):
        return list(self.layer2loc.keys())

    def layer2module(self, layer):
        """
        Get a PyTorch Module object according to the layer name.

        Parameter:
        ---------
        layer[str]: layer name

        Return:
        ------
        module[Module]: PyTorch Module object
        """
        module = self.model
        for k in self.layer2loc[layer]:
            module = module._modules[k]

        return module


class Vgg11(DNN):

    def __init__(self):
        super(Vgg11, self).__init__()

        self.model = tv_models.vgg11()
        self.model.load_state_dict(torch.load(
            pjoin(DNNBRAIN_MODEL, 'vgg11.pth')))
        self.layer2loc = {'conv1': ('features', '0'), 'conv1_relu': ('features', '1'),
                          'conv1_maxpool': ('features', '2'), 'conv2': ('features', '3'),
                          'conv2_relu': ('features', '4'), 'conv2_maxpool': ('features', '5'),
                          'conv3': ('features', '6'), 'conv3_relu': ('features', '7'),
                          'conv4': ('features', '8'), 'conv4_relu': ('features', '9'),
                          'conv4_maxpool': ('features', '10'), 'conv5': ('features', '11'),
                          'conv5_relu': ('features', '12'), 'conv6': ('features', '13'),
                          'conv6_relu': ('features', '14'), 'conv6_maxpool': ('features', '15'),
                          'conv7': ('features', '16'), 'conv7_relu': ('features', '17'),
                          'conv8': ('features', '18'), 'conv8_relu': ('features', '19'),
                          'conv8_maxpool': ('features', '20'), 'fc1': ('classifier', '0'),
                          'fc1_relu': ('classifier', '1'), 'fc2': ('classifier', '3'),
                          'fc2_relu': ('classifier', '4'), 'fc3': ('classifier', '6'), }
        self.img_size = (224, 224)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            normalize
        ])

    @property
    def layers(self):
        return list(self.layer2loc.keys())

    def layer2module(self, layer):
        """
        Get a PyTorch Module object according to the layer name.

        Parameter:
        ---------
        layer[str]: layer name

        Return:
        ------
        module[Module]: PyTorch Module object
        """
        module = self.model
        for k in self.layer2loc[layer]:
            module = module._modules[k]

        return module
