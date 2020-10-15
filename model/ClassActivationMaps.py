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
import pickle
from utils import postprocess
from classes import classes as class_labels
from torch.nn.functional import relu, softmax
from tqdm import tqdm


class ClassActivationMaps:

    def __init__(self, model, layer_name=None, method='gradcam'):
        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.method = method
        self.hooks = None
        self.interactive = False
        self.methods = ['cam', 'gradcam', 'gradcam++', 'scorecam']
        if method not in self.methods:
            raise ValueError('invalid method name,\
                 plese choose one of the following method: cam, gradcam, gradcam++, scorecam')
    
    def get_cam(self, t, class_index=None):

        if not self.hooks:
            self.hooks = set_hook(self.model, self.layer_name)
        
        if self.method == 'cam' : cam_map = self.cam(t, class_index)
        if self.method == 'gradcam' : cam_map = self.gradcam(t, class_index)
        if self.method == 'gradcam++' : cam_map = self.gradcamplus(t, class_index)
        if self.method == 'scorecam' : cam_map = self.scorecam(t, class_index)

        # Remove hooks from last layer just in case if you train your model after obtaining CAM
        if not self.interactive:
            for hook in self.hooks:
                hook.remove()
                self.hooks = None
        # resize it to the shape of input image
        cam_map = nn.functional.interpolate(cam_map, size=(t.shape[2], t.shape[3]), mode='bilinear', align_corners=False)
        
        # remove batch and channel dimention
        cam_map = cam_map.squeeze(0).squeeze(0)

        # Normalize
        cam_map -= cam_map.min()
        cam_map /= cam_map.max()

        return cam_map
    
    def show_cam(self, t, class_index=None):

        # specify size of the image
        plt.gcf().set_size_inches(8, 8)
        plt.axis('off')
        # plot CAM
        cam = self.get_cam(t, class_index)
        img = postprocess(t)
        # plot input image
        plt.imshow(cam, cmap='jet')
        plt.imshow(img, alpha=0.5)
        # save overlapping output
        plt.savefig(f'{self.method}.png', bbox_inches='tight')
        plt.show()
 
    
    def gradcam(self, t, class_index=None):

        # obtain prediction for the given image
        pred = self.model(t)
        print(f'predicted class : {pred.argmax().item()}')
        target = torch.zeros_like(pred)
        # if index is not provided use the class index with the largest logit value
        if not class_index:
            class_index = pred.argmax().item()
        print(f'showing cam for class : {class_index} {class_labels[class_index]}')
        target[0, class_index] = 1
        # allow gradients for target vector
        target.require_grad=True
        # obtain loss for the class "index"
        loss = torch.sum(pred*target)
        # remove previous gradients
        self.model.zero_grad()
        # obtain gradients
        loss.backward(retain_graph=True)
        # obtain graients for the last conv layer from the backward_hook
        grad = self.hooks[1].output[0].data
        # obtain weights for each feature map of last conv layer using gradients of that layer
        grad = torch.mean(grad, (2, 3), keepdim=True)
        # obtain output of last conv layer from the forward_hook
        conv_out = self.hooks[0].output.data
        # obtain weighed feature maps, keep possitive influence only
        cam = relu((conv_out * grad).sum(1, keepdim=True))

        return cam

    def cam(self, t, class_index):

        with torch.no_grad():
            pred = self.model(t)

        cam = self.hooks[0].output.data
        cam = cam.permute(0, 2, 3, 1)

        weight = list(self.model.named_children())[-1][1].weight.data

        weight = weight.t()

        cam = cam @ weight
        cam = cam.permute(0, 3, 1, 2)
        cam = torch.max(cam, dim=1, keepdim=True).values

        return cam

    def gradcamplus(self, t, class_index):

        pred = self.model(t)
        print(f'predicted class : {pred.argmax().item()}')
        target = torch.zeros_like(pred)
        # if index is not provided use the class index with the largest logit value
        if not class_index:
            class_index = pred.argmax().item()
        print(f'showing cam for class : {class_index} {class_labels[class_index]}')
        target[0, class_index] = 1
        # allow gradients for target vector
        target.require_grad=True
        # obtain loss for the class "index"
        loss = torch.exp(torch.sum(pred*target))
        # remove previous gradients
        self.model.zero_grad()
        # obtain gradients
        loss.backward(retain_graph=True)
        # obtain graients for the last conv layer from the backward_hook
        grad = self.hooks[1].output[0].data
        # get second order derivative (kind of)
        grad_2 = grad ** 2
        # get third order derivative (kind of)
        grad_3 = grad ** 3
        # get global average of gradients of each feature map
        grad_3_mul = torch.mean(grad, (2, 3), keepdim=True)
        # prepare for alpha denominator
        grad_3 = grad_3 * grad_3_mul
        # get alpha
        alpha = grad_2 / (2 * grad_2 + grad_3 + 1e-06)
        # get final weights of each feature map
        weight = (alpha * relu(grad)).sum((2, 3), keepdim=True)
        # obtain output of last conv layer from the forward_hook
        conv_out = self.hooks[0].output.data
        # obtain weighed feature maps, keep possitive influence only
        cam = relu(conv_out * weight).sum(1, keepdim=True)
    
        return cam

    def scorecam(self, t, class_index, bs=64):

        # obtain prediction of network
        pred = self.model(t)

        if not class_index:
            class_index = pred.argmax().item()

        # obtain output of last conv layer from forward_hook
        cam = self.hooks[0].output.data

        # mask input image using each feature map of last conv layer
        for index in tqdm(range(cam.shape[1])):

            # extract one feature map
            f_map = cam[:, index, :, :].unsqueeze(1)

            # resize it to match input image size
            f_map = nn.functional.interpolate(f_map, size=(t.shape[2], t.shape[3]), mode='bilinear', align_corners=False)

            # Normalize in range [0, 1]
            f_map -= f_map.min()
            f_map /= f_map.max()

            # obtain masked input image using resized and normalized feature map
            if index == 0:
                M = t*f_map
            else:
                M = torch.cat((M, t * f_map))
        
        # obtain prediction of model for each masked input image
        with torch.no_grad():
            for i in tqdm(range(M.shape[0]//bs)):
                if i == 0:
                    scores = self.model(M[bs*i:bs*(i+1),...])
                else:
                    scores = torch.cat((scores, self.model(M[bs*i:bs*(i+1),...])))

        
        # for each masked input image calculate the increase in logit value
        tscores = torch.tensor([i - pred[0, index].item() for i in scores[:, index]])
        # Normalize so that they sum to 1.
        tscores = softmax(tscores)
        # reshape to the size of [1, C, 1, 1]
        tscores = tscores.view(1, -1, 1, 1)
        # obtained weighted feature map using normalized scores
        cam = (relu(cam * tscores)).sum(1, keepdim=True)

        return cam

if __name__ == '__main__':
    model = torchvision.models.resnet18(pretrained=True)
    cam = ClassActivationMaps(model, method='gradcam')
    image = read_image(os.getcwd()+'/test/dogs.jpg')
    cam.show_cam(image)
        
        