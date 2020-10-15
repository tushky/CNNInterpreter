import os
import sys
sys.path.insert(1, './utils')
import torch
from hook import set_hook
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
from read_image import read_image, tensor_to_image
from sign_classification import SignNet
import pickle
from utils import postprocess
from classes import classes as class_labels

class GradCAMPlus(nn.Module):

    def __init__(self, cnn, name=None):

        '''
        Generate Gradient weighted Class Activation Map (GradCAM) using pretrained ConvNet.
        Pass pretrained ConvNet model and name of the last conv layer in the model.
        If name is not provided, we will recursivey search for last occuring conv layer in the model.
        '''
        
        super().__init__()

        # Pretrained model
        self.cnn = cnn

        self.cnn.eval()

        # name of last conv layer
        self.name = name

        self.relu = nn.ReLU()
        self.interactive = False
        self.hooks = None

    def get_cam(self, t, index=None):

        '''
        Takes input image tensor and returns class activation map of class "index"

        t : image tensor of shape [B, C, H, W]
        index : index of the class, if not provided we will select the class predicted by the model
        '''

        # assign forward and backward hook to the last conv layer of the model
        if not self.hooks:
            self.hooks = set_hook(self.cnn, self.name)
        # obtain prediction for the given image
        pred = self.cnn(t)
        print(f'predicted class : {pred.argmax().item()}')
        target = torch.zeros_like(pred)
        # if index is not provided use the class index with the largest logit value
        if not index:
            index = pred.argmax().item()
        print(f'showing cam for class : {index} {class_labels[index]}')
        target[0, index] = 1
        # allow gradients for target vector
        target.require_grad=True
        # obtain loss for the class "index"
        loss = torch.exp(torch.sum(pred*target))
        # remove previous gradients
        self.cnn.zero_grad()
        # obtain gradients
        loss.backward(retain_graph=True)
        # obtain graients for the last conv layer from the backward_hook
        grad = self.hooks[1].output[0].data
        # get second order derivative
        grad_2 = grad ** 2
        # get third order derivative
        grad_3 = grad ** 3
        # get global average of gradients of each feature map
        grad_3_mul = torch.mean(grad, (2, 3), keepdim=True)
        # prepare for alpha denominator
        grad_3 = grad_3 * grad_3_mul
        # get alpha
        alpha = grad_2 / (2 * grad_2 + grad_3 + 1e-06)
        # get final weights of each feature map
        weight = (alpha * self.relu(grad)).sum((2, 3), keepdim=True)
        # obtain output of last conv layer from the forward_hook
        conv_out = self.hooks[0].output.data
        # obtain weighed feature maps, keep possitive influence only
        cam = self.relu(conv_out * weight).sum(1, keepdim=True)
        # resize weighted feature maps to th size of input image
        cam = nn.functional.interpolate(cam, scale_factor=t.shape[2] // cam.shape[2], mode='bilinear', align_corners=False)
        # remove batch and channel dims
        cam = cam.squeeze(0).squeeze(0)
        # normalize CAM
        cam -= cam.min()
        cam /= cam.max()
        # Remove hooks from last layer just in case if you train your model after obtaining CAM
        if not self.interactive:
            for hook in self.hooks:
                hook.remove()
                self.hooks = None
        return cam

    def show_cam(self, t, index=None):

            # specify size of the image
            plt.gcf().set_size_inches(8, 8)
            plt.axis('off')
            # plot CAM
            cam = self.get_cam(t, index)
            img = postprocess(t)
            # plot input image
            plt.imshow(cam, cmap='jet')
            plt.imshow(img, alpha=0.5)
            # save overlapping output
            plt.savefig('grad_cam++.png', bbox_inches='tight')
            plt.show()
    
    def top_five(self, t):
        '''
        If you want to obtain CAM with respect to multiple classes you have to call get_cam with each class id
        which means forward pass will execute multiple times which i dont like. I need a better way to implement this.
        '''
        self.interactive = True
        plt.gcf().set_size_inches(8, 8)
        plt.axis('off')
        for index in [243, 282, 242, 281, 254]:
            cam = self.get_cam(t, index)
            img = postprocess(t)
            
            plt.imshow(cam, cmap='jet')
            plt.imshow(img, alpha=0.5)

            plt.savefig('grad_cam_plus.png', bbox_inches='tight')
            plt.show()


if __name__ == '__main__':

    cnn = torchvision.models.googlenet(pretrained=True)
    cam = GradCAMPlus(cnn)
    image = read_image(os.getcwd()+'/test/spider.png')
    cam.show_cam(image)