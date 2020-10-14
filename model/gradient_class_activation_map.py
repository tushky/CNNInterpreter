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

class GradCAM(nn.Module):

    def __init__(self, cnn, name=None):

        '''
        Generate Gradient weighted Class Activation Map (GradCAM) using pretrained ConvNet.
        Pass pretrained ConvNet model and name of the last conv layer in the model.
        If name is not provided, we will recursivey search for last occuring conv layer in the model.
        '''
        
        super().__init__()

        # Pretrained model
        self.cnn = cnn

        # name of last conv layer
        self.name = name

        self.relu = nn.ReLU()

    def get_cam(self, t, index=None):

        '''
        Takes input image tensor and returns class activation map of class "index"

        t : image tensor of shape [B, C, H, W]
        index : index of the class, if not provided we will select the class predicted by the model
        '''
        
        # put model in evaluation mode
        self.cnn.eval()

        # assign forward and backward hook to the last conv layer of the model
        self.hooks = set_hook(self.cnn, self.name)

        # obtain prediction for the given image
        pred = self.cnn(t)

        print(f'predicted class : {pred.argmax().item()}')

        target = torch.zeros_like(pred)

        # if index is not provided use the class index with the largest logit value
        if index:
            target[0, index] = 1
        else:
            target[0, pred.argmax().item()] = 1

        # allow gradients for target vector
        target.require_grad=True

        # obtain loss for the class "index"
        loss = torch.sum(pred*target)

        # remove previous gradients
        self.cnn.zero_grad()

        # obtain gradients
        loss.backward(retain_graph=True)

        # obtain graients for the last conv layer from the backward_hook
        grad = self.hooks[1].output[0].data

        # obtain weights for each feature map of last conv layer using gradients of that layer
        grad = torch.mean(grad, (2, 3), keepdim=True)

        # obtain output of last conv layer from the forward_hook
        conv_out = self.hooks[0].output.data

        # obtain weighed feature maps, keep possitive influence only
        cam = self.relu((conv_out * grad).sum(1, keepdim=True))

        # resize weighted feature maps to th size of input image
        cam = nn.functional.interpolate(cam, scale_factor=t.shape[2] // cam.shape[2], mode='bilinear', align_corners=False)

        # remove batch and channel dims
        cam = cam.squeeze(0).squeeze(0)

        # normalize CAM
        cam -= cam.min()
        cam /= cam.max()

        # Remove hooks from last layer
        for hook in self.hooks:
            hook.remove()

        return cam

    def show_cam(self, t, index=None):

            plt.gcf().set_size_inches(8, 8)
            plt.axis('off')

            cam = self.get_cam(t, index)
            img = postprocess(t)
            
            plt.imshow(cam, cmap='jet')
            plt.imshow(img, alpha=0.5)

            plt.savefig('grad_cam.png', bbox_inches='tight')
            plt.show()
    

if __name__ == '__main__':

    cnn = torchvision.models.resnet34(pretrained=True)
    cam = GradCAM(cnn)
    image = read_image(os.getcwd()+'/test/spider.png')
    cam.show_cam(image)