import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from scipy import ndimage
import os
import sys

sys.path.insert(1, './utils')
from read_image import read_image
from utils import postprocess
from hook import set_hook

class CAM(nn.Module):

    def __init__(self, cnn, name):
        
        super().__init__()
        self.cnn = cnn
        self.name = name
    
    def get_cam(self, image):


        self.hooks = set_hook(self.cnn, self.name)

        with torch.no_grad():
            pred = self.cnn(image)

        cam = self.hooks[0].output.data
        print('cam shape', cam.shape)
        #cam = cam.squeeze(0)
        cam = cam.permute(0, 2, 3, 1)

        weight = list(self.cnn.named_children())[-1][1].weight.data

        weight = weight.t()
        cam = cam @ weight
        cam = cam.permute(0, 3, 1, 2)
        cam = torch.max(cam, dim=1, keepdim=True).values
        cam = nn.functional.interpolate(cam, scale_factor=image.shape[2] // cam.shape[2], mode='bilinear', align_corners=False)
        print(cam.shape)
        cam = cam.squeeze(0).squeeze(0)

        for hook in self.hooks:
            hook.remove()

        return cam

    def show_cam(self, t):

            plt.gcf().set_size_inches(8, 8)
            plt.axis('off')

            cam = self.get_cam(t)
            img = postprocess(t)
            
            plt.imshow(cam, cmap='jet')
            plt.imshow(img, alpha=0.5)

            plt.savefig('cam.png', bbox_inches='tight')
            plt.show()

if __name__ == '__main__':

    cnn = torchvision.models.googlenet(pretrained=True)
    cam = CAM(cnn, 'inception5b')
    print(os.getcwd())
    image = read_image(os.getcwd()+'/test/spider.png')

    cam.show_cam(image)
