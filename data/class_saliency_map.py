import os
import sys

sys.path.insert(1, './utils')
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


class ClassModel(nn.Module):

        def __init__(self, cnn):

            super().__init__()

            self.cnn = cnn
            self.cnn.eval()
            for param in cnn.parameters() : param.requires_grad = False

        
        def forward(self, class_number, lr=0.01, lamda = 0.000005):

            x = np.zeros((1, 3, 244, 244))
            x += np.reshape([[0.485, 0.456, 0.406]], (1, 3, 1, 1))
            x = torch.Var(x, requires_grad=True, dtype=torch.float32)

            optimizer = optim.Adam([x], lr=lr)

            for i in tqdm(range(1000)):
                out = self.cnn(x)
                loss = lamda * torch.norm(x) - out[0, class_number]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return x

if __name__ == '__main__':
    alexnet = models.vgg16(pretrained=True)
    viz_class = ClassModel(alexnet)
    image = viz_class(177)
    image = tensor_to_image(image)
    image.save(os.getcwd()+'/test/maps/class_viz.jpg')