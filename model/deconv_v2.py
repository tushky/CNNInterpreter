import os
import sys

sys.path.insert(1, './utils')
import math
import torch
import torchvision
import numpy as np
from hook import Hook
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from read_image import read_image, tensor_to_image
from torchvision.utils import make_grid, save_image
from hook import Hook, set_hook


imagenet = models.vgg16(pretrained=True)
imagenet.eval()
image_path = os.getcwd() + '/test/cat_dog.png'
image = read_image(image_path, 'imagenet')
image.requires_grad=True

conv_out = {}
def get_output(output):
    print(output.shape)
    conv_out[0] = output.detach()
    return output


def get_input(input):

    t = conv_out[0]
    print(f't shape : {t.shape}')
    max_activation = [None] * t.shape[1]

    # For each of the feature map find the maximum activation
    for kernal in range(t.shape[1]):
        max_activation[kernal] = (kernal, torch.max(t[0, kernal, :, :]).item(),
            torch.argmax(t[0, kernal, :, :]).item())

    # Sort features according to their maximum activation value    
    max_activation.sort(key=lambda x: x[1], reverse=True)

    # Construct "num_kernel" number of input tensor for deconv model
    dconv_input = torch.zeros_like(t)
    index = np.unravel_index(max_activation[0][2], (t.shape[2], t.shape[3]))
    dconv_input[0, max_activation[0][0], index] = t[0, max_activation[0][0], index]
    #dconv_input[0, max_activation[i][0], index] = 1.0
    return (dconv_input, input[1], input[2])

hooks = set_hook(imagenet, backward_input_fn = get_input, forward_output_fn = get_output)

out = imagenet(image)
loss = 0 - torch.sum(out)
print(loss)
loss.backward()
print(hooks)
for hook in hooks:
    hook.remove()
plt.imshow(torchvision.utils.make_grid(image.grad.detach(), normalize=True).permute(1, 2, 0))
plt.show()