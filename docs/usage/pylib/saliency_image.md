# Saliency Image
There are some examples of getting saliency images through using python library of DNNBrain.

The original image used in this doc is displayed as below:
<center>![original](../../img/n02108551_26574.JPEG)</center>
## Example 1
Vanilla Saliency Image
```
from PIL import Image
from dnnbrain.dnn.base import ip
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.algo import VanillaSaliencyImage

# Prepare DNN and image
dnn = AlexNet()
image = Image.open('n02108551_26574.JPEG')

# Get the vanilla saliency image of the image
# which displays the saliency parts that contribute to 
# the activation of the 276th unit of fc3.
vanilla = VanillaSaliencyImage(dnn)
vanilla.set_layer('fc3', 276)
img_out = vanilla.backprop(image)

# transform to PIL image and save out
img_out = ip.to_pil(img_out, True)
img_out.save('n02108551_26574_vanilla_saliency.JPEG')
```
The vanilla saliency image is displayed as below:
<center>![vanilla](../../img/n02108551_26574_vanilla_saliency.JPEG)</center>

## Example 2
Guided Saliency Image
```
from PIL import Image
from dnnbrain.dnn.base import ip
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.algo import GuidedSaliencyImage

# Prepare DNN and image
dnn = AlexNet()
image = Image.open('n02108551_26574.JPEG')

# Get the guided saliency image of the image
# which displays the saliency parts that contribute to 
# the activation of the 276th unit of fc3.
guided = GuidedSaliencyImage(dnn)
guided.set_layer('fc3', 276)
img_out = guided.backprop(image)

# transform to PIL image and save out
img_out = ip.to_pil(img_out, True)
img_out.save('n02108551_26574_guided_saliency.JPEG')
```
The guided saliency image is displayed as below:
<center>![guided](../../img/n02108551_26574_guided_saliency.JPEG)</center>
