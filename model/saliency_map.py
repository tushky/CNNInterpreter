import os
import sys
sys.path.insert(1, './utils')
import torch.nn as nn
from torchvision import models
from read_image import read_image
import matplotlib.pyplot as plt


class SaliencyMap(nn.Module):

    def __init__(self, cnn):

        super().__init__()

        self.cnn = cnn
        self.cnn.eval()
        for param in self.cnn.parameters() : param.requires_grad=False
    
    def get_map(self, t):

        t.requires_grad = True

        out = self.cnn(t)

        loss = out.max()

        loss.backward()

        s_map = t.grad.detach().squeeze(0)

        s_map = s_map.abs().max(0).values

        s_map -= s_map.min()
        s_map /= s_map.max()

        return s_map
    
    def show_map(self, t):

        s_map = self.get_map(t)
        plt.axis('off')
        plt.imshow(s_map, cmap='gray')
        plt.show()



if __name__ == '__main__':

    image_path = os.getcwd() + '/test/horse5.jpg'
    output_path = os.getcwd() + '/test/maps/output.jpg'
    image = read_image(image_path)
    alexnet = models.resnet34(pretrained=True)
    net = SaliencyMap(alexnet)
    net.show_map(image)
