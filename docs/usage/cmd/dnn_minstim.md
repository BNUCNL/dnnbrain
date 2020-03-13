# Name
dnn_minstim - Simplify a stimulus into a minimal part which could cause equivalent activation as the original stimlus using DNN

# Synopsis
dnn_minstim -net Net -layer Layer -chn Channel -stim Stimulus -out Output

# Arguments
## Required Arguments
|Argument|Discription|
|--------|-----------|
|net     |Name of a neural network|
|stim    |Path of a .stim.csv file which contains stimulus information|
|layer   |Name of the target layer.</br>Only support one layer each time.</br>E.g., 'conv1' represents the first convolution layer, and 'fc1' represents the first full connection layer.|
|chn     |Index of target channel.</br>Only support one channel each time.</br>Channel index starts from 1.|
|out     |Output path to save the simplfied image|

# Outputs
A series of minimal images based on your stimilus and interested net information

# Examples
```
dnn_minstim -net AlexNet -layer conv1 -chn 6 -stim ./faces.stim.csv -out ./image/minstim/
```

