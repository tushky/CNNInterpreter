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
import matplotlib.pyplot as plt
from read_image import read_image, tensor_to_image
from torchvision.utils import make_grid, save_image

class DeconvNet(nn.Module):

    def __init__(self, cnn, layer_num):

        super().__init__()

        self.cnn = cnn
        self.cnn.eval()
        self.layer_num = layer_num
        self.construct_convnet()
    
    def construct_convnet(self):

        self.conv_model = OrderedDict()
        self.dconv_model = OrderedDict()

        i = 0
        
        num_conv, num_relu, num_pool = 1, 1, 1

        while i <= self.layer_num:

            layer = self.cnn.features[i]

            if isinstance(layer,  nn.Conv2d):
                self.conv_model[f'conv_{num_conv}'] = layer
                dconv_layer= nn.ConvTranspose2d(layer.out_channels, layer.in_channels, layer.kernel_size, layer.stride, layer.padding)
                dconv_layer.weight.data = layer.weight.data
                self.dconv_model[f'dconv_{num_conv}'] = dconv_layer
                num_conv += 1

            if isinstance(layer, nn.ReLU):
                self.conv_model[f'relu_{num_relu}'] = layer
                self.dconv_model[f'unrelu_{num_relu}'] = layer
                num_relu += 1

            if isinstance(layer, nn.MaxPool2d):
                layer.return_indices = True
                maxunpool = nn.MaxUnpool2d(layer.kernel_size, layer.stride, layer.padding)
                self.conv_model[f'pool_{num_pool}'] = layer
                self.dconv_model[f'unpool_{num_pool}'] = maxunpool
                num_pool += 1
            i += 1
        
        self.conv_model = nn.Sequential(self.conv_model)
        self.dconv_model = nn.Sequential(OrderedDict(reversed(list(self.dconv_model.items()))))

    def conv_forward(self, t):

        num_layer = 1
        self.pool_indices = {}
        for layer in self.conv_model:
            if isinstance(layer, nn.MaxPool2d):
                t, self.pool_indices[f'unpool_{num_layer}'] = layer(t)
                num_layer += 1
            else:
                t = layer(t)
        return t
    
    def guided_conv_forward(self, t):

        pool_layer = 1
        relu_layer = 1

        self.pool_indices = {}
        self.relu_indices = {}

        for layer in self.conv_model:

            if isinstance(layer, nn.MaxPool2d):
                t, self.pool_indices[f'unpool_{pool_layer}'] = layer(t)
                pool_layer += 1

            elif isinstance(layer, nn.ReLU):
                t = layer(t)
                self.relu_indices[f'unrelu_{relu_layer}'] = torch.where(t > 0, torch.tensor([1]), torch.tensor([0]))
                relu_layer += 1

            else:
                t = layer(t)
                
        return t
    
    def dconv_forward(self, t):

        num_layer = 1
        for key, layer in self.dconv_model.named_children():
            if isinstance(layer, nn.MaxUnpool2d):
                t = layer(t, self.pool_indices[key])
                num_layer += 1
            else:
                t = layer(t)
        return t
    
    def guided_dconv_forward(self, t):

        pool_layer = 1
        relu_layer = 1

        for key, layer in self.dconv_model.named_children():

            if isinstance(layer, nn.MaxUnpool2d):
                t = layer(t, self.pool_indices[key])
                pool_layer += 1
            
            elif isinstance(layer, nn.ReLU):
                t = layer(t)
                t *= self.relu_indices[key]
                relu_layer += 1

            else:
                t = layer(t)
        return t

    def construct_dconv_input(self, t, num_kernel=9):

        assert len(t.shape) == 4, f"Incorrect input shape, expected tensor of shape 4, recived tensor of shape {t.shape}"
        max_activation = [None] * t.shape[1]

        for kernal in range(t.shape[1]):
            max_activation[kernal] = (kernal, torch.max(t[0, kernal, :, :]).item(), torch.argmax(t[0, kernal, :, :]).item())
            
        max_activation.sort(key=lambda x: x[1], reverse=True)

        for i in range(num_kernel):
            dconv_input = torch.zeros_like(t)
            index = np.unravel_index(max_activation[i][2], (t.shape[2], t.shape[3]))
            dconv_input[0, max_activation[i][0], index] = t[0, max_activation[i][0], index]
            #dconv_input[0, max_activation[i][0], index] = 1.0
            yield dconv_input
    
    def forward(self, t, path, num_kernel):

        t = self.conv_forward(t)
        for i, dconv_input in tqdm(enumerate(self.construct_dconv_input(t, num_kernel))):
            out = self.dconv_forward(dconv_input)
            out_img = tensor_to_image(out, True)
            if i == 0 : maps = out_img.unsqueeze(0)
            else : maps = torch.cat((maps, out_img.unsqueeze(0)))

        return save_image(maps, fp=path, nrow=int(math.sqrt(num_kernel)))
    
    def guided_forward(self, t, path, num_kernel=9):
        t = self.guided_conv_forward(t)
        for i, dconv_input in tqdm(enumerate(self.construct_dconv_input(t, num_kernel))):
            out = self.guided_dconv_forward(dconv_input)
            out_img = tensor_to_image(out, True)
            if i == 0 : maps = out_img.unsqueeze(0)
            else : maps = torch.cat((maps, out_img.unsqueeze(0)))

        return make_grid(maps, nrow=int(math.sqrt(num_kernel)))

    def prediction(self, t):
        
        for layer in self.cnn.features:

            if isinstance(layer, nn.MaxPool2d) and layer.return_indices:
                t, _ = layer(t)
            else:
                t = layer(t)
        t = self.cnn.avgpool(t)
        t = t.view(1, -1)
        t = self.cnn.classifier(t)
        return t.argmax().item()



if __name__ == '__main__' :

    imagenet = models.alexnet(pretrained=True)
    image_path = os.getcwd() + '/test/cat_dog.png'
    deconv_output_path = os.getcwd() + '/test/maps/output_dconv.jpg'
    guided_deconv_output_path = os.getcwd() + '/test/maps/output_guided_dconv.jpg'
    image = read_image(image_path, 'imagenet')
    deconvnet = DeconvNet(imagenet, 6)
    out = deconvnet.guided_forward(image, deconv_output_path)
    plt.imshow(out.permute(1, 2, 0).numpy())
    plt.show()
