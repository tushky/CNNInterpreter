import os
import math
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict
from read_image import read_image, tensor_to_image
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt


class DeepInvert(nn.Module):

    def __init__(self, cnn, layer_number):

        super().__init__()
        self.cnn = cnn
        self.layer_number = layer_number
        self.cnn.eval()
    
    def forward(self, x):
        #x = self.cnn.features(x)
        #x = self.cnn.avgpool(x)
        #x = torch.flatten(x, 1)
        for name, layer in self.cnn.features.named_children():
            x = layer(x)
            if name == self.layer_number : break
        return x


if __name__ == '__main__':
    image = read_image(os.getcwd() + '/test/dog.jpg')
    cnn = models.alexnet(pretrained=True)
    net = DeepInvert(cnn, layer_number='4')
    probs = net(image)
    print(probs)
    print(probs.shape)