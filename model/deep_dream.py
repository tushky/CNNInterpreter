import os
import math
import torch
import copy
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from read_image import read_image, tensor_to_image
from torchvision.utils import make_grid, save_image
from scipy.ndimage import gaussian_filter
class DeepDream(nn.Module):

    def __init__(self, cnn, level_number):

        super().__init__()
        self.cnn = cnn
        for param in self.cnn.parameters() : param.requires_grad = False
        self.level_number = level_number

    def forward(self, x):

        x.requires_grad = True

        for i in tqdm(range(100)):
            x.requires_grad = True

            t = x

            for name, layer in self.cnn.named_children():
                t = layer(t)
                if name == str(self.level_number):
                    break

            loss = torch.norm(t)
            loss.backward()
            gradients = x.grad.detach().numpy()[0]
            gradients = np.transpose(gradients, (1, 2, 0))
            gradients/= np.std(gradients) + 1e-08
            sigma = (i * 4.0) / 10.0 + 0.5
            grad1 = gaussian_filter(gradients, sigma=sigma)
            grad2 = gaussian_filter(gradients, sigma=sigma*2)
            grad3 = gaussian_filter(gradients, sigma=sigma*0.5)
            grad = (grad1 + grad2 + grad3)
            print(f"Grad min : {grad.min()}, max : {grad.max()}")
            grad = np.transpose(grad, (2, 0, 1))
            x = x.detach()
            x[0,:] += 0.1 * grad
            x = torch.clamp(x, min=-3, max=3)

        return x


if __name__ == '__main__':
    cnn = models.vgg16(pretrained=True)
    cnn = cnn.features
    #print(cnn)
    image_path = os.getcwd() + '/test/sky.jpg'
    output_path = os.getcwd() + '/test/maps/output.jpg'
    image = read_image(image_path, 'alexnet')
    model = DeepDream(cnn, 30)
    out = model(image)
    out = tensor_to_image(out)
    plt.imshow(out)
    plt.show()
