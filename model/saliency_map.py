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


class SaliencyMap(nn.Module):

    def __init__(self, cnn):

        super().__init__()

        self.cnn = cnn
        self.cnn.eval()
        for param in self.cnn.parameters() : param.requires_grad=False
    
    def forward(self, t):

        t.requires_grad = True
        out = self.cnn(t)
        loss = out.max()
        print(loss)
        loss.backward()
        t_grad = t.grad.detach().squeeze(0)
        t_grad = t_grad.abs().max(0).values
        return t_grad.numpy()

if __name__ == '__main__':

    image_path = os.getcwd() + '/test/horse5.jpg'
    output_path = os.getcwd() + '/test/maps/output.jpg'

    alexnet = models.vgg13(pretrained=True)
    net = SaliencyMap(alexnet)
    image = read_image(image_path)
    saliency_map = net(image)
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    saliency_map = Image.fromarray((saliency_map*255).astype(np.uint8))
    saliency_map.save(output_path)
    plt.imshow(saliency_map, cmap='gray')
    plt.show()