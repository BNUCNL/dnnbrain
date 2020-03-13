## Name
<b>dnn_act</b> - extract activation of stimuli from DNN.  
The dnn_act takes as principal input the network name and the path of the file containing stimuli imformation, and produces activation of the stimuli in the network as output. Further specific requirments like roi selection, pooling method and GPU using can be asserted by optional inputs.

## Synopsis
dnn_act -net Net [-layer Layer [Layer ...]] [-chn Channel [Channel ...]] [-dmask DnnMask] -stim Stimulus [-pool Pooling] [-cuda] -out Output

## Arguments
### Required Arguments
|Argument|Discription|
|--------|-----------|
|net     |Name of a neural network|
|stim    |Path of a .stim.csv file which contains stimulus information|
|out     |Output path with a suffix as .act.h5|

### Optional Arguments
|Argument|Discription|
|--------|-----------|
|layer   |Name of the target layer(s).</br>Default is all.</br>E.g., 'conv1' represents the first convolution layer, and 'fc1' represents the first full connection layer.|
|chn     |Index of target channel(s).</br>Default is all.</br>Channel index starts from 1.|
|dmask   |Path of a .dmask.csv file in which detailed information of neuron(s) of interest in DNN is specified.</br>Argument layer/chn and dmask are mutually exclusive. Provide only one of them if needed. |
|pool    |Pooling method.</br>Default is none.</br>max: max pooling; mean: mean pooling; median: median pooling; L1: 1-norm; L2: 2-norm.|
|cuda    |Use GPU|

## Outputs
An .act.h5 file containing the extracted activation that can be read and saved with the module dnnbrain.io.fileio.ActivationFile.

## Examples
These examples demonstrate the activation extraction function. Activation from target layers of stimuli provided by test.stim.csv was extracted and maxpooled, finally saved in the test.act.h5 file.  

```
# Asserting target layers using the -layer argument
dnn_act -net AlexNet -layer conv1 conv5_relu fc2_relu -stim ./test.stim.csv -pool max -out ./test.act.h5
```

```
# Asserting target layers using the -dmask argument
dnn_act -net AlexNet -dmask ./test.dmask.csv -stim ./test.stim.csv -pool max -out ./test.act.h5
```