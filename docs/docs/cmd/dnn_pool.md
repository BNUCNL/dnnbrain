## Name
<b>dnn_pool</b> - Pooling DNN activation for each channel.

## Synopsis
dnn_pool -act Activation [-layer Layer [Layer ...]] [-chn Channel [Channel ...]] [-dmask DnnMask] -meth Method -out Output

## Arguments
### Required Arguments
|Argument|Discription|
|--------|-----------|
|act    |Path of a .act.h5 file which contains activation information to extract features.|
|meth     |Method of activation pooling.</br>Choices are 'max', 'mean', 'median', 'L1', 'L2'.</br>The L1 means 1-norm and L2 means 2-norm.|
|out     |Output path with a suffix as .act.h5.|

### Optional Arguments
|Argument|Discription|
|--------|-----------|
|layer   |Name of the target layer(s).</br>Default is all.</br>E.g., 'conv1' represents the first convolution layer, and 'fc1' represents the first full connection layer.|
|chn     |Index of target channel(s).</br>Default is all.</br>Channel index starts from 1.|
|dmask   |Path of a .dmask.csv file in which detailed information of neuron(s) of interest in DNN is specified.</br>Argument layer/chn and dmask are mutually exclusive. Provide only one of them if needed. |

## Outputs
An .act.h5 file containing the pooling activation that can be read and saved with the module dnnbrain.io.fileio.ActivationFile.

## Examples
These examples demonstrate the activation pooling extraction function. Activation provided by test.act.h5 was pooling and finally saved in the dnn_fe.act.h5 file.  

```
# Asserting target layers using the -layer argument
dnn_pool -act ./test.act.h5 -layer conv5 -meth max -out ./dnn_pool1.act.h5
```

```
# Asserting target layers using the -dmask argument
dnn_pool -act ./test.act.h5 -dmask ./test.dmask.csv -meth max -out ./dnn_pool2.act.h5
```

```
# Not asserting layers which means target layers and chns are all
dnn_pool -act ./test.act.h5 -meth max -out ./dnn_pool3.act.h5
```