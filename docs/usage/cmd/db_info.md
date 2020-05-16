## Name
<b>db_info</b> - Provide important information of specific dnn or file

## Synopsis
db_info filename/netname

## Arguments
### Required Arguments
|Argument|Discription|
|--------|-----------|
|name     |Filename or Netname.</br>Filename only support suffix with '.stim.csv','.dmask.csv','.act.h5','.roi.h5','.rdm.h5'.</br>Remember to run this command in the right path or it can't find your file.</br>Netname only support 'AlexNet', 'VggFace', 'Vgg11'|

## Check for dnn information
Provide dnn's structure parameters and layer to location information
### Examples
```
db_info AlexNet
```
The output information is displayed as below:
<center>![view](../../img/AlexNet_info.png)</center>

## Check for file information
### .stim.csv
Provide information of stimuli type, path, data types and number of stimuli.
#### Examples
```
db_info test.stim.csv
```
The output information is displayed as below:
<center>![view](../../img/stim_info.png)</center>

### .dmask.csv
Provide information of mask layer, chn, row and column.
#### Examples
```
db_info test.dmask.csv
```
The output information is displayed as below:
<center>![view](../../img/dmask_info.png)</center>

### .act.h5
Provide activation shape in different layers and its statistical information.
#### Examples
```
db_info test.act.h5
```
The output information is displayed as below:
<center>![view](../../img/act_info.png)</center>

### .roi.h5
Provide ROI names, data shape of brain response and its statistical information.
#### Examples
```
db_info test.roi.h5
```
The output information is displayed as below:
<center>![view](../../img/roi_info.png)</center>

### .rdm.h5
Provide representation distance matrices (RDMs) shape for DNN activation and brain activation and their statistical information.
#### Examples
```
# Checking information for RDM type of brain
db_info brain.rdm.h5
```
The output information is displayed as below:
<center>![view](../../img/brain_rdm_info.png)</center>
```
# Checking information for RDM type of dnn
db_info dnn.rdm.h5
```
The output information is displayed as below:
<center>![view](../../img/dnn_rdm_info.png)</center>