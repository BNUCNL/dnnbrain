# Layer to Module
DNNBrain allows users to conveniently approach to the module inside a built-in DNN by a commonly used layer name. For each built-in DNN, its available layer names and the mapping between layer names and modules are shown as follow. And we take the AlexNet as example to interpret mechanism of the mapping.

## AlexNet
Generally, we can approach to a module by its location in the DNN framework. For example, according to AlexNet framework (Fig. 1, left), we can locate the first convolutional layer by its location (features, 0). However, DNNBrain users can easily do it by the layer name 'conv1' according to the mapping (Fig. 1, right).
<center>
	<img src="../../img/layer2module/alexnet_framework.png" width=65%><img src="../../img/layer2module/alexnet_mapping.png" width=35%>
	Figure 1
</center>

## VggFace
<center>
	<img src="../../img/layer2module/vggface_framework.png" width=65%><img src="../../img/layer2module/vggface_mapping.png" width=35%>
	Figure 2
</center>

## Vgg11
<center>
	<img src="../../img/layer2module/vgg11_framework.png" width=65%><img src="../../img/layer2module/vgg11_mapping.png" width=35%>
	Figure 3
</center>
