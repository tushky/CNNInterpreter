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
from tqdm import tqdm
import sys

sys.path.insert(1, './utils')
from read_image import read_image
from utils import postprocess
from hook import set_hook

class ScoreCAM(nn.Module):

    def __init__(self, cnn, name=None):
        
        super().__init__()
        self.cnn = cnn
        self.name = name
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)


    def prediction(self, img, index):
        self.cnn.eval()
        pred = cnn(img)
        return pred[0, index].item()

    def get_cam(self, img, index=None, bs=64):

        # put model in evaluation mode
        self.cnn.eval()

        # set hooks on last conv layer
        self.hooks = set_hook(self.cnn, self.name)

        # obtain prediction of network
        pred = self.cnn(img)

        if not index:
            index = pred.argmax().item()

        # obtain output of last conv layer from forward_hook
        cam = self.hooks[0].output.data

        # mask input image using each feature map of last conv layer
        for index in tqdm(range(cam.shape[1])):

            # extract one feature map
            f_map = cam[:, index, :, :].unsqueeze(1)

            # resize it to match input image size
            f_map = nn.functional.interpolate(f_map, scale_factor=img.shape[2]//f_map.shape[2], mode='bilinear', align_corners=False)

            # Normalize in range [0, 1]
            f_map -= f_map.min()
            f_map /= f_map.max()

            # obtain masked input image using resized and normalized feature map
            if index == 0:
                M = img*f_map
            else:
                M = torch.cat((M, img * f_map))
        
        # obtain prediction of model for each masked input image
        with torch.no_grad():
            for i in tqdm(range(M.shape[0]//bs)):
                if i == 0:
                    scores = cnn(M[bs*i:bs*(i+1),...])
                else:
                    scores = torch.cat((scores, cnn(M[bs*i:bs*(i+1),...])))

        
        # for each masked input image calculate the increase in logit value
        tscores = torch.tensor([i - pred[0, index].item() for i in scores[:, index]])

        # Normalize so that they sum to 1.
        tscores = self.softmax(tscores)

        # reshape to the size of [1, C, 1, 1]
        tscores = tscores.view(1, -1, 1, 1)

        # obtained weighted feature map using normalized scores
        cam = (self.relu(cam * tscores)).sum(1, keepdim=True)

        # resize it to the shape of input image
        cam = nn.functional.interpolate(cam, scale_factor=img.shape[2] // cam.shape[2], mode='bilinear', align_corners=False)
        
        # remove batch and channel dimention
        cam = cam.squeeze(0).squeeze(0)

        # Normalize
        cam -= cam.min()
        cam /= cam.max()

        # remove hooks
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

            plt.savefig('score_cam.png', bbox_inches='tight')
            plt.show()

if __name__ == '__main__':

    cnn = torchvision.models.resnet34(pretrained=True)
    cam = ScoreCAM(cnn)
    print(os.getcwd())
    image = read_image(os.getcwd()+'/test/spider.png')
    cam.show_cam(image, 281)