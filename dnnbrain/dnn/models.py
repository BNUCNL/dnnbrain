import os
import copy
import time
import torch
import numpy as np

from os.path import join as pjoin
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision import models as tv_models
from dnnbrain.dnn.core import Stimulus, Activation
from dnnbrain.dnn.base import ImageSet, VideoSet, dnn_mask, array_statistic

DNNBRAIN_MODEL = pjoin(os.environ['DNNBRAIN_DATA'], 'models')


def dnn_truncate(netloader, layer):
    """
    Truncate the neural network at the specified convolution layer.
    Notice that all truncated models were consisted of Sequential, which may differ from the orginal model.

    Parameters:
    -----------
    netloader[NetLoader]: a neural network netloader, initialized from NetLoader in io module
    layer[str]: truncated layer.

    Returns:
    --------
    truncated_net[torch.nn.Sequential]: truncated model.
    """
    assert netloader.model is not None, "Please define netloader by calling NetLoader from module io"
    assert netloader.layer2indices is not None, "Please define netloader by calling NetLoader from module io"
    indices = netloader.layer2indices[layer]
    prefc_indices = netloader.layer2indices['prefc']
    model_frame = nn.Sequential(*netloader.model.children())
    truncate_model = _get_truncate_layers(model_frame, indices)
    if 'fc' in layer:
        # Re-define forward method
        def forward(x):
            x = truncate_model[:prefc_indices[0]](x)
            x = torch.flatten(x, 1)
            x = truncate_model[prefc_indices[0]:](x)
            return x
        truncate_model.forward = forward
    return truncate_model


def _get_truncate_layers(model_frame, indices):
    """
    Subfunction of dnn_truncate to access truncated model recursively.
    """
    if len(indices) > 1:
        parent_sequential = nn.Sequential(*model_frame[:indices[0]].children())
        append_sequential = nn.Sequential(*model_frame[indices[0]].children())
        truncate_model = nn.Sequential(*parent_sequential, _get_truncate_layers(append_sequential, indices[1:]))
    else:
        truncate_model = nn.Sequential(*model_frame[:(indices[0]+1)])
    return truncate_model


def dnn_train_model(dataloaders_train, model, criterion, optimizer, num_epoches, train_method='tradition',
                    dataloaders_train_test=None, dataloaders_val_test=None):
    """
    Function to train a DNN model

    Parameters:
    ------------
    dataloaders_train[dataloader]: dataloader of traindata to train
    dataloaders_train_test[dataloader]: dataloader of traindata to test
    dataloaders_val_test[dataloader]: dataloader of validationdata to test
    model[class/nn.Module]: DNN model without pretrained parameters
    criterion[class]: criterion function
    optimizer[class]: optimizer function
    num_epoches[int]: epoch times.
    train_method[str]: training method, by default is 'tradition'. 
                       For some specific models (e.g. inception), loss needs to be calculated in another way.
                       
    Returns:
    --------
    model[class/nn.Module]: model with trained parameters.
    metric_dict: If dataloaders_train_test and dataloaders_val_test are not None:
                 epoch      loss          ACC_train_top1, ACC_train_top5, ACC_val_top1, ACC_val_top5
                  {1: (2.144788321990967,     0.2834,         0.8578,        0.2876,       0.8595),
                   2: (1.821894842262268,     0.45592,        0.91876,       0.4659,       0.9199),
                   3: (1.6810704930877685,    0.50844,        0.9434,        0.5012,       0.9431)}
                  
                 If dataloaders_train_test and dataloaders_val_test are None:
                 epoch      loss
                  {1: (2.144788321990967),
                   2: (1.821894842262268),
                   3: (1.6810704930877685)}
                
    """
    warnings.filterwarnings("ignore")
    LOSS = []
    ACC_train_top1 = []
    ACC_train_top5 = []
    ACC_val_top1 = []
    ACC_val_top5 = []
    EPOCH = []
    
    time0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)
            
    for epoch in range(num_epoches):
        EPOCH.append(epoch+1)
        print('Epoch time {}/{}'.format(epoch+1, num_epoches))
        print('-'*10)
        time1 = time.time()
        running_loss = 0.0
        
        for inputs, targets in dataloaders_train:
            inputs.requires_grad_(True)
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if train_method == 'tradition':
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                elif train_method == 'inception':
                    # Google inception model
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, targets)
                    loss2 = criterion(aux_outputs, targets)
                    loss = loss1 + 0.4*loss2
                else:
                    raise Exception('Not Support this method yet, please contact authors for implementation.')
                
                _, pred = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
                
            # Statistics loss in every batch
            running_loss += loss.item() * inputs.size(0)
        
        # Caculate loss in every epoch
        epoch_loss = running_loss / len(dataloaders_train.dataset)
        print('Loss: {}\n'.format(epoch_loss))
        LOSS.append(epoch_loss)
        
        # Caculate ACC_train every epoch
        if dataloaders_train_test:
            model_copy = copy.deepcopy(model)
            _, _, train_acc_top1, train_acc_top5 = dnn_test_model(dataloaders_train_test, model_copy)
            print('top1_acc_train: {}\n'.format(train_acc_top1))
            print('top5_acc_train: {}\n'.format(train_acc_top5))
            ACC_train_top1.append(train_acc_top1)
            ACC_train_top5.append(train_acc_top5)
    
        # Caculate ACC_val every epoch
        if dataloaders_val_test:
            model_copy = copy.deepcopy(model)
            _, _, val_acc_top1, val_acc_top5 = dnn_test_model(dataloaders_val_test, model_copy)
            print('top1_acc_test: {}\n'.format(val_acc_top1))
            print('top5_acc_test: {}\n'.format(val_acc_top5))
            ACC_val_top1.append(val_acc_top1)
            ACC_val_top5.append(val_acc_top5)
        
        #print time of a epoch
        time_epoch = time.time() - time1
        print('This epoch training complete in {:.0f}m {:.0f}s'.format(time_epoch // 60, time_epoch % 60))
    
    # store LOSS, ACC_train, ACC_val to a dict
    if dataloaders_train_test and dataloaders_val_test:
        metric = zip(LOSS, ACC_train_top1, ACC_train_top5, ACC_val_top1, ACC_val_top5)
        metric_dict = dict(zip(EPOCH, metric))
    else:
        metric_dict = dict(zip(EPOCH, LOSS))
    
    time_elapsed = time.time() - time0
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, metric_dict


def dnn_test_model(dataloaders, model):
    """
    Test model accuracy.
    
    Parameters:
    -----------
    dataloaders[dataloader]: dataloader generated from dataloader(PicDataset)
    model[class/nn.Module]: DNN model with pretrained parameters
    
    Returns:
    --------
    model_target[array]: model output
    actual_target [array]: actual target
    test_acc_top1[float]: prediction accuracy of top1
    test_acc_top5[float]: prediction accuracy of top5
    """ 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    model_target = []
    model_target_top5 = []
    actual_target = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloaders):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, outputs_label = torch.max(outputs, 1)
            outputs_label_top5 = torch.topk(outputs, 5)
            
            model_target.extend(outputs_label.cpu().numpy())
            model_target_top5.extend(outputs_label_top5[1].cpu().numpy())
            actual_target.extend(targets.numpy())
            
    model_target = np.array(model_target)
    model_target_top5 = np.array(model_target_top5)
    actual_target = np.array(actual_target)
    
    # Caculate the top1 acc and top5 acc
    test_acc_top1 = 1.0*np.sum(model_target == actual_target)/len(actual_target)
    
    test_acc_top5 = 0.0
    for i in [0,1,2,3,4]:
        test_acc_top5 += 1.0*np.sum(model_target_top5.T[i]==actual_target)
    test_acc_top5 = test_acc_top5/len(actual_target)
    
    return model_target, actual_target, test_acc_top1, test_acc_top5


class GraftLayer(nn.Module):
    def __init__(self, netloader, layer, fc_out_num, channel=None, feature_extract=True):
        """
        Connect the truncated_net to a full connection layer.

        Parameters:
        -----------
        netloader[NetLoader]: a neural network netloader, initialized from NetLoader in io module.
        layer[str]: truncated layer
        fc_out_num[int]: the number of the out_features of the full connection layer
        channel[iterator]: The indices of out_channels of the selected convolution layer
        feature_extract[bool]: If feature_extract = False, the model is finetuned and all model parameters are updated.
            If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
            https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        """
        super(GraftLayer, self).__init__()
        self.truncate_net = dnn_truncate(netloader, layer)
        if feature_extract:
            for param in self.truncate_net.parameters():
                param.requires_grad = False

        # Set a test data to get output size of the truncate_net
        indicator_val = torch.randn((1,3,netloader.img_size[0],netloader.img_size[1]))
        indicator_output = self.truncate_net(indicator_val)
        
        if channel is not None:
            assert 'conv' in layer, "Selected channel only happened in convolution layer."
        self.channel = channel
        # fc input number
        fc_in_num = indicator_output.view(indicator_output.size(0),-1).shape[-1]
        self.fc = nn.Linear(fc_in_num, fc_out_num)
    
    def forward(self, x):
        x = self.truncate_net(x)
        # if self.channel is not None :
        # I'm not quite sure how to trained a model with some channels but not all.
        # Therefore channel is not be finished.
        # The original one is
        # if self.channel is not None:
        #   x = x[:, self.channel]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TransferredNet(nn.Module):
    def __init__(self, truncated_net, fc_in_num, fc_out_num, channel=None, feature_extract=True):
        """
        Connect the truncated_net to a full connection layer.

        Parameters:
        -----------
        truncated_net[torch.nn.Module]: a truncated neural network from the pretrained network
        fc_in_num[int]: the number of the in_features of the full connection layer
        fc_out_num[int]: the number of the out_features of the full connection layer
        channel[iterator]: The indices of out_channels of the selected convolution layer
        feature_extract[bool]: If feature_extract = False, the model is finetuned and all model parameters are updated.
            If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
            https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        """
        super(TransferredNet, self).__init__()
        self.truncated_net = truncated_net
        if feature_extract:
            for param in self.truncated_net.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(fc_in_num, fc_out_num)
        self.channel = channel

    def forward(self, x):
        x = self.truncated_net(x)
        if self.channel is not None:
            # extract the specified channel's output
            x = x[:, self.channel]
        x = x.view(x.size(0), -1)  # (batch_num, unit_num)
        x = self.fc(x)
        return x
        

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
    """Deep neural network"""

    def __init__(self):

        self.model = None
        self.layer2loc = None
        self.img_size = None

    def save(self, fname):
        """
        Save DNN parameters

        Parameter:
        ---------
        fname[str]: output file name with suffix as .pth
        """
        assert fname.endswith('.pth'), 'File suffix must be .pth'
        torch.save(self.model.state_dict(), fname)

    def set(self, model, parameters=None, layer2loc=None, img_size=None):
        """
        Load DNN model, parameters, layer2loc and img_size manually

        Parameters:
        ----------
        model[nn.Modules]: DNN model
        parameters[state_dict]: Parameters of DNN model
        layer2loc[dict]: map layer name to its location in the DNN model
        img_size[tuple]: the input image size
        """
        self.model = model
        if parameters is not None:
            self.model.load_state_dict(parameters)
        self.layer2loc = layer2loc
        self.img_size = img_size

    def compute_activation(self, stimuli, dmask, pool_method=None):
        """
        Extract DNN activation

        Parameters:
        ----------
        stimuli[Stimulus|ndarray]: input stimuli
            If is Stimulus, loaded from files on the disk.
            If is ndarray, its shape is (n_stim, n_chn, height, width)
        dmask[Mask]: The mask includes layers/channels/rows/columns of interest.
        pool_method[str]: pooling method, choices=(max, mean, median, L1, L2)

        Return:
        ------
        activation[Activation]: DNN activation
        """
        # prepare stimuli loader
        transform = Compose([Resize(self.img_size), ToTensor()])
        if isinstance(stimuli, np.ndarray):
            stim_set = [Image.fromarray(arr.transpose((1, 2, 0))) for arr in stimuli]
            stim_set = [(transform(img), 0) for img in stim_set]
        elif isinstance(stimuli, Stimulus):
            if stimuli.meta['type'] == 'image':
                stim_set = ImageSet(stimuli.meta['path'], stimuli.get('stimID'), transform=transform)
            elif stimuli.meta['type'] == 'video':
                stim_set = VideoSet(stimuli.meta['path'], stimuli.get('stimID'), transform=transform)
            else:
                raise TypeError('{} is not a supported stimulus type.'.format(stimuli.meta['type']))
        else:
            raise TypeError('The input stimuli must be an instance of Tensor or Stimulus!')
        data_loader = DataLoader(stim_set, 8, shuffle=False)

        # -extract activation-
        # change to eval mode
        self.model.eval()
        n_stim = len(stim_set)
        activation = Activation()
        for layer in dmask.layers:
            # prepare dnn activation hook
            acts_holder = []

            def hook_act(module, input, output):

                # copy activation
                acts = output.detach().numpy().copy()
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

            module = self.model
            for k in self.layer2loc[layer]:
                module = module._modules[k]
            hook_handle = module.register_forward_hook(hook_act)

            # extract DNN activation
            for stims, _ in data_loader:
                # stimuli with shape as (n_stim, n_chn, height, width)
                self.model(stims)
                print('Extracted activation of {0}: {1}/{2}'.format(
                    layer, len(acts_holder), n_stim))
            activation.set(layer, np.asarray(acts_holder))

            hook_handle.remove()

        return activation

    def get_kernel(self, layer, kernel_num=None):
        """
        Get kernel's weights of the layer

        Parameters:
        ----------
        layer[str]: layer name
        kernel_num[int]: the sequence number of the kernel

        Return:
        ------
        kernel[array]: kernel weights
        """
        # localize the module
        module = self.model
        for k in self.layer2loc[layer]:
            module = module._modules[k]

        # get the weights
        kernel = module.weight
        if kernel_num is not None:
            kernel = kernel[kernel_num]

        return kernel.detach().numpy()

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
        module = self.model
        for k in self.layer2loc[layer]:
            module = module._modules[k]

        # ablate kernels' weights
        if channels is None:
            module.weight.data[:] = 0
        else:
            channels = [chn - 1 for chn in channels]
            module.weight.data[channels] = 0

    def train(self, data, n_epoch, criterion, optimizer=None, method='tradition', target=None):
        """
        Train the DNN model

        Parameters:
        ----------
        data[Stimulus|ndarray]: training data
            If is Stimulus, load stimuli from files on the disk.
                Note, the data of the 'label' item in the Stimulus object will be used as
                output of the model when 'target' is None.
            If is ndarray, it contains stimuli with shape as (n_stim, n_chn, height, width).
                Note, the output data must be specified by 'target' parameter.
        n_epoch[int]: the number of epochs
        criterion[str|object]: criterion function
            If is str, choices=('classification', 'regression').
            If is not str, it must be torch loss object.
        optimizer[object]: optimizer function
            If is None, use Adam default.
            If is not None, it must be torch optimizer object.
        method[str]: training method, by default is 'tradition'.
            For some specific models (e.g. inception), loss needs to be calculated in another way.
        target[ndarray]: the output of the model
            Its shape is (n_stim,) for classification or (n_stim, n_feat) for regression.
                Note, n_feat is the number of features of the last layer.
        """
        # prepare data loader
        transform = Compose([Resize(self.img_size), ToTensor()])
        if isinstance(data, np.ndarray):
            stim_set = [Image.fromarray(arr.transpose((1, 2, 0))) for arr in data]
            stim_set = [(transform(img), trg) for img, trg in zip(stim_set, target)]
        elif isinstance(data, Stimulus):
            if data.meta['type'] == 'image':
                stim_set = ImageSet(data.meta['path'], data.get('stimID'),
                                    data.get('label'), transform=transform)
            elif data.meta['type'] == 'video':
                stim_set = VideoSet(data.meta['path'], data.get('stimID'),
                                    data.get('label'), transform=transform)
            else:
                raise TypeError(f"{data.meta['type']} is not a supported stimulus type.")

            if target is not None:
                # We presume small quantity stimuli will be used in this way.
                # Usually hundreds or thousands such as fMRI stimuli.
                stim_set = [(img, trg) for img, trg in zip(stim_set[:][0], target)]
        else:
            raise TypeError('The input data must be an instance of Tensor or Stimulus!')
        data_loader = DataLoader(stim_set, 8, shuffle=False)

        # prepare criterion
        if criterion == 'classification':
            criterion = nn.CrossEntropyLoss()
        elif criterion == 'regression':
            criterion = nn.MSELoss()

        # prepare optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        loss_list = []
        time1 = time.time()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.train()
        self.model = self.model.to(device)
        for epoch in range(n_epoch):
            print(f'Epoch-{epoch+1}/{n_epoch}')
            print('-' * 10)
            time2 = time.time()
            running_loss = 0.0

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
                running_loss += loss.item() * inputs.size(0)

            # calculate loss in every epoch
            epoch_loss = running_loss / len(data_loader.dataset)
            print(f'Loss: {epoch_loss}')
            loss_list.append(epoch_loss)

            # print time of a epoch
            epoch_time = time.time() - time2
            print('This epoch costs {:.0f}m {:.0f}s\n'.format(epoch_time // 60, epoch_time % 60))

        time_elapsed = time.time() - time1
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


class AlexNet(DNN):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.model = tv_models.alexnet()
        self.model.load_state_dict(torch.load(
            pjoin(DNNBRAIN_MODEL, 'alexnet_param.pth')))
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


class VggFace(DNN):

    def __init__(self):
        super(VggFace, self).__init__()

        self.model = VggFaceModel()
        self.model.load_state_dict(torch.load(
            pjoin(DNNBRAIN_MODEL, 'vgg_face_dag.pth')))
        self.layer2loc = None
        self.img_size = (224, 224)


class Vgg11(DNN):

    def __init__(self):
        super(Vgg11, self).__init__()

        self.model = tv_models.vgg11()
        self.model.load_state_dict(torch.load(
            pjoin(DNNBRAIN_MODEL, 'vgg11_param.pth')))
        self.layer2loc = None
        self.img_size = (224, 224)


if __name__ == '__main__':
    dnn = AlexNet()
    stim = Stimulus('e:/useful_things/data/AI/dnnbrain_data/test/image/sub-CSI1_ses-01_imagenet.stim.csv')
    dnn.train(stim, 2, 'classification')
