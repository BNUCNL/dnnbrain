import copy
import time
import torch
import numpy as np
from torch import nn


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
        

class Vgg_face(nn.Module):
    """Vgg_face's model architecture"""

    def __init__(self):
        super(Vgg_face, self).__init__()
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
