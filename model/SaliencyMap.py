import os
import sys
sys.path.insert(1, './utils')
from utils import postprocess
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from read_image import read_image
import matplotlib.pyplot as plt
#import torchvision.transforms as transforms
from torchvision.utils import make_grid


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.5):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    
class SaliencyMap(nn.Module):

    def __init__(self, cnn):

        super().__init__()

        self.cnn = cnn
        self.cnn.eval()
        for param in self.cnn.parameters() : param.requires_grad=False
    
    def get_map(self, t):

        t.requires_grad = True

        out = self.cnn(t)
        indices = out.argmax(dim=1)
        print(indices)
        #indices = indices 
        target = torch.zeros_like(out)
        target[:, indices] = 1.
        loss = torch.sum(out * target)

        loss.backward()
        
        s_map = t.grad.detach().mean(dim=0)
        print(s_map.min(), s_map.max())
        s_map = torch.clamp(s_map, min=-5, max=5)
        s_map = s_map.abs().max(0).values

        s_map -= s_map.min()
        s_map /= (s_map.max()+1e-05)

        return s_map
    
    def show_map(self, t):

        s_map = self.get_map(t)
        plt.axis('off')
        plt.imshow(s_map, cmap='gray')
        plt.show()



if __name__ == '__main__':

    image_path = os.getcwd() + '/test/ante.jpeg'
    output_path = os.getcwd() + '/test/maps/output.jpg'
    image = read_image(image_path)
    add_noise = AddGaussianNoise()
    out = add_noise(image)
    images = []
    for i in range(25):
        images.append(add_noise(image))
    images = torch.cat(images, dim=0)
    #print(img.shape)
    alexnet = models.googlenet(pretrained=True)
    net = SaliencyMap(alexnet)
    net.show_map(images)
    #print(saliency_map)
    #img_times_grad = saliency_map.unsqueeze(0).unsqueeze(0) * image
    #plt.imshow(postprocess(img_times_grad))
    #plt.show()
    #print(saliency_map.shape)
