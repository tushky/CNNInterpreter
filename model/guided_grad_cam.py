import os
import math
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict
from read_image import read_image, tensor_to_image
from torchvision.utils import make_grid, save_image
from gradient_class_activation_map import GradCAM
from deconvolution_network import DeconvNet
import matplotlib.pyplot as plt


cnn = torchvision.models.vgg16(pretrained=True)
image = read_image(os.getcwd()+'/test/spider.png')

grad_cam = GradCAM(cnn)
cam = grad_cam.get_cam(image)

plt.imshow(cam, cmap='jet')
#plt.show()
print(cam.shape)

deconv = DeconvNet(cnn, 25)
deconv_out = deconv.guided_forward(image)
out = deconv_out * cam.unsqueeze(0)
out = out.permute(1, 2, 0)
print(out.shape)
plt.imshow(out)
plt.show()
