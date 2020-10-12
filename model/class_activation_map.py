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

class Hook:
    
    def __init__(self, name, layer, backward=False):
        
        self.name = name
        
        if backward:
            #print('backward pass')
            layer.register_backward_hook(self.hook_fn)
        else:
            layer.register_forward_hook(self.hook_fn)
            
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        self.module = module
        print(f'{self.name} output shape : {output.shape}')


class CAM(nn.Module):

    def __init__(self, cnn, name):
        
        super().__init__()
        self.cnn = cnn
        self.name = name

        self.preprocess = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                            transforms.Lambda(lambda t: t.unsqueeze(0))
                            ])

        self.recursive_hook(self.cnn, self.name)

    
    def get_cam(self, image):
        img = self.preprocess(image)
        pred = self.cnn(img)

        cam = self.hook.output.data
        #cam = F.interpolate(cam, scale_factor=32, mode='bicubic', align_corners=False)
        cam = cam.squeeze(0)
        cam = cam.permute(1, 2, 0)
        print(cam.shape)

        weight = list(self.cnn.named_children())[-1][1].weight.data
        weight = weight.t()
        print(weight.shape)

        out = torch.matmul(cam, weight)
        out = out.numpy()
        print(out.shape)

        mean_out = np.max(out, axis=2)
        mean_out = ndimage.zoom(mean_out, 32)
        print(mean_out.shape)

        plt.imshow(mean_out, cmap='jet')
        test_img = img.squeeze(0).permute(1, 2, 0).numpy()
        plt.imshow((test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img)), alpha=0.5)
        plt.show()

    def recursive_hook(self, module, target_name):
        for name, layer in module.named_children():
            if name == target_name:
                print(f'{name} layer hooked')
                self.hook = Hook(name, layer)
            self.recursive_hook(layer, target_name)

if __name__ == '__main__':

    cnn = torchvision.models.googlenet(pretrained=True)

    cam = CAM(cnn, 'inception5b')
    print(os.getcwd())
    image = Image.open(os.getcwd()+'/test/catdog.jpeg').convert('RGB')

    cam.get_cam(image)
