# Name
<b>dnn_saliency</b> - Find the saliency parts of an image that contributes to the activation of the specified channel.  
# Synopsis
```
dnn_saliency [-h] -net Net -layer Layer -chn Channel [Channel ...]
             -stim Stimulus [-meth Method] [-mode Mode] [-cmap Colormap]
			 [-vmin Vmin] [-vmax Vmax] [-show] [-out Output]
```
# Arguments
## Required Arguments
|Argument|Discription|
|--------|-----------|
|net     |a neural network name|
|layer   |specify the layer</br>E.g., 'conv1' represents the first convolution layer and 'fc1' represents the first full connection layer.|
|chn     |Channel numbers used to specify which channels are used to find salience images|
|stim    |a .stim.csv file which contains stimulus information|

## Optional Arguments
|Argument|Discription|
|--------|-----------|
|meth    |the method used to generate the saliency image.</br>choose from ('guided', 'vanilla'). Default is 'guided'|
|mode    |Visualization mode of the saliency image.</br>RGB: visualize derivatives directly;</br>gray: retain the maximal magnitude of RGB channels for each pixel, and visualize as a gray image.</br>Note: -cmap, -vmin and -vmax options are only valid at the gray mode.</br>choose from ('RGB', 'gray'). Default is 'RGB'.|
|cmap    |show salience images with the specified colormap</br>choose from matplotlib colormaps. Default is coolwarm.|
|vmin    |The minimal value used in colormap is applied in all salience images.</br>Default is the minimal value of each salience image for itself.|
|vmax    |The maximal value used in colormap is applied in all salience images.</br>Default is the maximal value of each salience image for itself.|
|show    |If used, display stimuli and salience images in figures.|
|out     |an output directory where the figures are saved|

# Outputs
Display or save out the figures which contain stimuli and their salience images corresponding each channel.

# Examples
Display examples' salience images corresponding to the 294<sup>th</sup> and 23<sup>rd</sup> channels in layer 'fc3'.
```
dnn_saliency -net AlexNet -layer fc3 -chn 294 23 -stim examples.stim.csv -show
```
Save out to the current directory for examples' salience images corresponding to the 294<sup>th</sup> and 23<sup>rd</sup> channels in layer 'fc3'. Using gray mode and gray colormap.
```
dnn_saliency -net AlexNet -layer fc3 -chn 294 23 -stim examples.stim.csv -mode gray -cmap gray -out .
```