import torch
from torch import nn
import numpy as np
import time


def dnn_truncate(net, indices, layer):
    """
    truncate the neural network at the specified convolution layer

    Parameters:
    -----------
    net[torch.nn.Module]: a neural network
    indices[iterator]: a sequence of raw indices to find a layer

    Returns:
    --------
    truncated_net[torch.nn.Sequential]
    """
    if 'conv' in layer:
        if len(indices) > 1:
            tmp  = list(net.children())[:indices[0]]
            next = list(net.children())[indices[0]]
            actmodel = nn.Sequential(*(tmp+[dnn_truncate(next, indices[1:],layer='conv')]))
        elif len(indices) == 1:
            actmodel = nn.Sequential(*list(net.children())[:indices[0]+1])
        else:
            raise ValueError("The network has no this layer.")
    elif 'fc' in layer:
        actmodel = net
        new_classifier = nn.Sequential(*list(net.children())[-1][:indices[1] + 1])
        if hasattr(actmodel, 'classifier'):
            actmodel.classifier = new_classifier
        elif hasattr(actmodel, 'fc'):
            actmodel.fc = new_classifier
        else:
            raise Exception("The network has no this layer.")
    else:
        raise Exception("The input of layer format is not right,input just like conv5 or fc1")

    return actmodel


class DNN2BrainNet(nn.Module):

    def __init__(self, truncated_net, channel_unit_num, fc_out_num, channel=None):
        """
        Connect the truncated_net to a full connection layer.

        Parameters:
        -----------
        truncated_net[torch.nn.Module]: a truncated neural network from the pretrained network
        channel_unit_num[int]: the number of units of each channel of the last layer in the truncated network
        fc_out_num[int]: the number of the out channels of the full connection layer
        channel[iterator]: The sequence numbers of out channels of the selected layer.
        """
        super(DNN2BrainNet, self).__init__()
        self.truncated_net = truncated_net
        if channel is None:
            channel_num = list(self.truncated_net.modules())[-1].out_channels
        else:
            channel_num = len(channel)
        self.fc = nn.Linear(channel_num * channel_unit_num, fc_out_num)
        self.channel = channel

    def forward(self, x):
        x = self.truncated_net(x)
        if self.channel is not None:
            # extract the specified channel's output
            x = x[:, self.channel]
        x = x.view(x.size(0), -1)  # (batch_num, unit_num)
        x = self.fc(x)
        return x

def dnn_finetuning(netloader,layer,out_class):
    """Fine-tuning the neural network, modifying the 1000 classifier to n classifier
    Parameters:
    -----------
    netloader[Netloader]: class netloader
    layer[str]: fully connected layer name of a DNN network
    out_features[str]:  specify the class of the new dnn model output
    Returns:
    --------
    dnn_model[torchvision.models]: the reconstructed dnn model.
    """
    if 'fc' not in layer:
        raise ValueError('Fine tuning only support to reconstruct fully connected layer')

    dnn_model = netloader.model
    # freeze parameters of pretrained network.
    for param in dnn_model.parameters():
        param.requires_grad = False
    # reconstruct the fully connected layer
    indices = netloader.layer2indices[layer]
    old_classifier = list(netloader.model.children())[-1][:indices[1]]
    in_features = list(netloader.model.children())[-1][indices[1]].in_features
    new_classifier = nn.Sequential(*(list(old_classifier) + [nn.Linear(in_features,out_class)]))

    if hasattr(dnn_model,'classifier'):
        dnn_model.classifier = new_classifier
    elif hasattr(dnn_model,'fc'):
        dnn_model.fc = new_classifier
    return dnn_model

def dnn_train_model(dataloaders, model, criterion, optimizer, num_epoches=200, train_method='tradition'):
    """
    Function to train a DNN model

    Parameters:
    ------------
    dataloaders[dataloader]: dataloader generated from dataloader(PicDataset)
    model[class/nn.Module]: DNN model without pretrained parameters
    criterion[class]: criterion function
    optimizer[class]: optimizer function
    num_epoches[int]: epoch times, by default is 200.
    train_method[str]: training method, by default is 'tradition'. 
                       For some specific models (e.g. inception), loss needs to be calculated in another way.
                       
    Returns:
    --------
    model[class/nn.Module]: model with trained parameters.
    """
    time0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)
    for epoch in range(num_epoches):
        print('Epoch time {}/{}'.format(epoch+1, num_epoches))
        print('-'*10)
        running_loss = 0.0
        running_correct = 0
        
        for _, inputs, targets in dataloaders:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(1):
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
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(pred==targets.data)
        
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_correct.double() / len(dataloaders.dataset)
        print('Loss: {.4f} Acc: {.4f}'.format(phase, epoch_loss, epoch_acc))
    time_elapsed =  time.time() - time0
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {.4f}'.format(epoch_acc))
    return model
    

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
    test_acc[float]: prediction accuracy 
    """ 
    time0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    model_target = []
    actual_target = []
    with torch.no_grad():
        for i, (_, inputs, targets) in enumerate(dataloaders):
            print('Now loading batch {}'.format(i+1))
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, outputs_label = torch.max(outputs, 1)
            model_target.extend(outputs_label.cpu().numpy())
            actual_target.extend(targets.numpy())
    model_target = np.array(model_target)
    actual_target = np.array(actual_target)
    test_acc = 1.0*np.sum(model_target == actual_target)/len(actual_target)
    time_elapsed =  time.time() - time0
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model_target, actual_target, test_acc
        
    
