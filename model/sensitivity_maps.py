"""
Created by : Tushar Gadhiya
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from utils import read_image, intergrated_inputs, noisy_inputs



class SensitivityMaps:

    """
    Collection of sensitivity based methods
    """

    def __init__(self, cnn):

        super().__init__()

        self.cnn = cnn
        self.cnn.eval()

    def smooth_grad(self, image, num_imgs=50, std=.8, colored=True, batch_size=64):

        """
            implimentation of SmoothGrad
        """

        images = noisy_inputs(image, std, num_imgs)
        dataloader = DataLoader(images, batch_size)
        gradients = []
        for imgs in dataloader:
            gradients.append(self.get_grad(imgs))
        gradients = torch.cat(gradients, dim=0)
        gradients = gradients.mean(dim=0, keepdim=True)
        gradients = self.normalize_grad(gradients, colored)
        return gradients

    def integrated_grad(self, image, num_steps=100, baseline=0, colored=True, batch_size=64):

        """
        Implimentation of integrated gradient method
        """

        images = intergrated_inputs(image, num_steps, baseline)
        dataloader = DataLoader(images, batch_size=batch_size)
        gradients = []
        for imgs in dataloader:
            gradients.append(self.get_grad(imgs))
        gradients = torch.cat(gradients, dim=0)
        gradients = gradients.mean(dim=0, keepdim=True)
        gradients = gradients * (image - baseline)
        gradients = self.normalize_grad(gradients, colored)
        return gradients

    def saliency_map(self, image, colored):

        """
            Implimnetation of venilla saliency/sensiticity map
        """

        grad = self.get_grad(image)
        return self.normalize_grad(grad, colored)

    def get_grad(self, image):

        """
            Obtain gradients of score with respect to input image
        """

        image.requires_grad = True

        out = self.cnn(image)

        indices = out.argmax(dim=1)
        print(indices)

        target = torch.zeros_like(out)
        target[:, 92] = 1.
        loss = torch.sum(out * target)

        loss.backward()

        return image.grad.detach()


    def normalize_grad(self, grad, colored):

        """
            Normalizes gradients
        """

        if not colored:
            grad = grad[0].abs().max(0).values
        else:
            grad = grad.abs()
        grad_max = np.percentile(grad.numpy(), 99)
        grad_min = grad.min().item()
        grad -= grad_min
        grad /= grad_max
        grad = torch.clamp(grad, min=0, max=1)
        return grad


    def get_map(self, image, method='smooth', colored=False):

        """
            Return sensitivity map of selected method
        """
        if method == 'saliency':
            return self.saliency_map(image, colored=colored)
        elif method == 'integrated':
            return self.integrated_grad(image, colored=colored)
        elif method == 'smooth':
            return self.smooth_grad(image, colored=colored)
        else:
            raise ValueError('Invalid method name {method}')


    def show_map(self, image, method='smooth', colored=False, path=None):

        """
            Display sensitivity map of selected method
        """
        s_map = self.get_map(image, method, colored)

        plt.axis('off')
        if colored:
            plt.imshow(s_map[0].permute(1, 2, 0), cmap='gray')
        else:
            plt.imshow(s_map, cmap='gray')
        if path:
            try:
                plt.savefig(path, bbox_inches='tight')
            except ValueError:
                print(f'invalid path {path}, unable to save file')
        #plt.show()
        plt.cla()


if __name__ == '__main__':

    image_path = os.getcwd() + '/images/snake.jpg'
    output_path = os.getcwd() + '/images/maps/output.jpg'
    img = read_image(image_path)
    model = models.vgg16(pretrained=True)
    net = SensitivityMaps(model)
    net.show_map(img, method='smooth', colored=False)
