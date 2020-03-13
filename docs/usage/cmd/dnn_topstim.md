# Name
dnn_topstim - For specific layer and channel in DNN network, select topK stimulus from a set of stimulus.

# Synopsis
dnn_topstim -model Net -stim Stimulus -dmask MaskFile -out Output -top K

# Arguments
## Required Arguments
|Argument|Discription|
|--------|-----------|
|model     |Name of a neural network|
|stim    |Path of a .stim.csv file which contains stimulus information|
|dmask   |Path of a .dmask.csv file which contains channels and layers information|
|top     |TOP K images that you want to present|
|out     |Output path to save the simplfied image|

# Outputs
several stim.csv file for further usage.

# Examples
```
dnn_topstim -model AlexNet -dmask ./faces.dmask.csv -top 6 -stim ./faces.stim.csv -out ./image/top_stim/
```
