import os
import copy
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

    def set_model(self, model, parameters=None, layer2loc=None, img_size=None):
        """
        Set DNN model, parameters, layer2loc and img_size manually

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

    def set_transform(self, train_transform=None, test_transform=None):
        """
        Set transform

        Parameters:
        ----------
        train_transform[torchvision.transform]:
            the transform used in training state
        test_transform[torchvision.transform]:
            the transform used in testing state
        """
        if train_transform is not None:
            self.train_transform = train_transform
        if test_transform is not None:
            self.test_transform = test_transform

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
        if isinstance(stimuli, np.ndarray):
            stim_set = [Image.fromarray(arr.transpose((1, 2, 0))) for arr in stimuli]
            stim_set = [(self.test_transform(img), 0) for img in stim_set]
        elif isinstance(stimuli, Stimulus):
            if stimuli.meta['type'] == 'image':
                stim_set = ImageSet(stimuli.meta['path'], stimuli.get('stimID'),
                                    transform=self.test_transform)
            elif stimuli.meta['type'] == 'video':
                stim_set = VideoSet(stimuli.meta['path'], stimuli.get('stimID'),
                                    transform=self.test_transform)
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

            module = self.layer2module(layer)
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
        # get the module
        module = self.layer2module(layer)

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
        module = self.layer2module(layer)

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
        if isinstance(data, np.ndarray):
            stim_set = [Image.fromarray(arr.transpose((1, 2, 0))) for arr in data]
            stim_set = [(self.train_transform(img), trg) for img, trg in zip(stim_set, target)]
        elif isinstance(data, Stimulus):
            if data.meta['type'] == 'image':
                stim_set = ImageSet(data.meta['path'], data.get('stimID'),
                                    data.get('label'), transform=self.train_transform)
            elif data.meta['type'] == 'video':
                stim_set = VideoSet(data.meta['path'], data.get('stimID'),
                                    data.get('label'), transform=self.train_transform)
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

        # start train
        loss_list = []
        time1 = time.time()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.train()
        model = self.model.to(device)
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
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    elif method == 'inception':
                        # Google inception model
                        outputs, aux_outputs = model(inputs)
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

    def test(self, data, task, target=None):
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

        Returns:
        -------
        test_dict[dict]:
            if task == 'classification':
                pred_value[array]: prediction values by the model
                true_value[array]: observation values
                acc_top1[float]: prediction accuracy of top1
                acc_top5[float]: prediction accuracy of top5
            if task == 'regression':
                pred_value[array]: prediction values by the model
                true_value[array]: observation values
                r_square[float]: R square between pred_values and true_values
        """
        # prepare data loader
        if isinstance(data, np.ndarray):
            stim_set = [Image.fromarray(arr.transpose((1, 2, 0))) for arr in data]
            stim_set = [(self.test_transform(img), trg) for img, trg in zip(stim_set, target)]
        elif isinstance(data, Stimulus):
            if data.meta['type'] == 'image':
                stim_set = ImageSet(data.meta['path'], data.get('stimID'),
                                    data.get('label'), transform=self.test_transform)
            elif data.meta['type'] == 'video':
                stim_set = VideoSet(data.meta['path'], data.get('stimID'),
                                    data.get('label'), transform=self.test_transform)
            else:
                raise TypeError(f"{data.meta['type']} is not a supported stimulus type.")

            if target is not None:
                # We presume small quantity stimuli will be used in this way.
                # Usually hundreds or thousands such as fMRI stimuli.
                stim_set = [(img, trg) for img, trg in zip(stim_set[:][0], target)]
        else:
            raise TypeError('The input data must be an instance of Tensor or Stimulus!')
        data_loader = DataLoader(stim_set, 8, shuffle=False)

        # start test
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        model = self.model.to(device)
        pred_values = []
        true_values = []
        if task == 'classification':
            pred_values_top5 = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)

                # collect outputs
                if task == 'classification':
                    _, pred_labels = torch.max(outputs, 1)
                    _, pred_labels_top5 = torch.topk(outputs, 5)
                    pred_values.extend(pred_labels.detach().numpy())
                    pred_values_top5.extend(pred_labels_top5.detach().numpy())
                    true_values.extend(targets.numpy())
                elif task == 'regression':
                    pred_values.extend(outputs.detach().numpy())
                    true_values.extend(targets.numpy())
                else:
                    raise ValueError('unsupported task:', task)

        test_dict = dict()
        pred_values = np.array(pred_values)
        true_values = np.array(true_values)
        if task == 'classification':
            pred_values_top5 = np.array(pred_values_top5)

            # calculate the top1 acc and top5 acc
            acc_top1 = np.sum(pred_values == true_values) / len(true_values)
            acc_top5 = 0.0
            for i in range(5):
                acc_top5 += np.sum(pred_values_top5[:, i] == true_values)
            acc_top5 = acc_top5 / len(true_values)

            test_dict['pred_value'] = pred_values
            test_dict['true_value'] = true_values
            test_dict['acc_top1'] = acc_top1
            test_dict['act_top5'] = acc_top5
        else:
            # calculate r_square
            r, _ = pearsonr(pred_values.ravel(), true_values.ravel())
            r_square = r ** 2

            test_dict['pred_value'] = pred_values
            test_dict['true_value'] = true_values
            test_dict['r_square'] = r_square

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
        self.layer2loc = {'conv1_1': ('features', '0'), 'conv1_1_relu': ('features', '1'),
                          'conv1_2': ('features', '2'), 'conv1_2_relu': ('features', '3'),
                          'conv1_maxpool': ('features', '4'), 'conv2_1': ('features', '5'),
                          'conv2_1_relu': ('features', '6'), 'conv2_2': ('features', '7'),
                          'conv2_2_relu': ('features', '8'), 'conv2_maxpool': ('features', '9'),
                          'conv3_1': ('features', '10'), 'conv3_1_relu': ('features', '11'),
                          'conv3_2': ('features', '12'), 'conv3_2_relu': ('features', '13'),
                          'conv3_3': ('features', '14'), 'conv3_3_relu': ('features', '15'),
                          'conv3_maxpool': ('features', '16'), 'conv4_1': ('features', '17'),
                          'conv4_1_relu': ('features', '18'), 'conv4_2': ('features', '19'),
                          'conv4_2_relu': ('features', '20'), 'conv4_3': ('features', '21'),
                          'conv4_3_relu': ('features', '22'), 'conv4_maxpool': ('features', '23'),
                          'conv5_1': ('features', '24'), 'conv5_1_relu': ('features', '25'),
                          'conv5_2': ('features', '26'), 'conv5_2_relu': ('features', '27'),
                          'conv5_3': ('features', '28'), 'conv5_3_relu': ('features', '29'),
                          'conv5_maxpool': ('features', '30'), 'fc6': ('classifier', '0'),
                          'relu6': ('classifier', '1'), 'fc7': ('classifier', '3'),
                          'relu7': ('classifier', '4'), 'fc8': ('classifier', '6'), }
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
            pjoin(DNNBRAIN_MODEL, 'vgg11_param.pth')))
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
