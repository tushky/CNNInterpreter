"""
Created by : Tushar Gadhiya
"""

import os
import torch
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from utils import read_image

add_noise = lambda t, std: t + torch.randn(t.size()) * std

class SensitivityMaps:

    """
    Collection of sensitivity based methods
    """

    def __init__(self, cnn):

        super().__init__()

        self.cnn = cnn
        self.cnn.eval()

    def smooth_grad(self, image, num_imgs=50, std=.8, colored=True):

        """
            implimentation of SmoothGrad
        """
        images = []

        for _ in range(num_imgs):
            images.append(add_noise(image, std))
        images = torch.cat(images, dim=0)

        grad = self.get_grad(images)
        grad = grad.mean(dim=0, keepdim=True)
        grad = self.normalize_grad(grad, colored)
        return grad

    def integrated_grad(self, image, num_steps=100, colored=True):

        """
        Implimentation of integrated gradient method
        """

        images = []

        for i in range(num_steps):
            images.append(image * (i / num_steps))

        images = torch.cat(images, dim=0)

        grad = self.get_grad(images)
        #grad = grad * images.detach()
        grad = grad.mean(dim=0, keepdim=True)
        grad = grad * image
        grad = self.normalize_grad(grad, colored)
        return grad

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


    def show_map(self, image, method='smooth', colored=False):

        """
            Display sensitivity map of selected method
        """
        s_map = self.get_map(image, method, colored)

        plt.axis('off')
        if colored:
            plt.imshow(s_map[0].permute(1, 2, 0), cmap='gray')
        else:
            plt.imshow(s_map, cmap='gray')
        plt.show()


if __name__ == '__main__':

    image_path = os.getcwd() + '/test/ante.jpeg'
    output_path = os.getcwd() + '/test/maps/output.jpg'
    img = read_image(image_path)
    model = models.vgg16(pretrained=True)
    net = SensitivityMaps(model)
    net.show_map(img, method='integrated', colored=True)
