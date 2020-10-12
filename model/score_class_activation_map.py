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
        #print(f'{self.name} output shape : {output.shape}')

class GradCAM(nn.Module):

    def __init__(self, cnn, name=None):
        
        super().__init__()
        self.cnn = cnn
        self.name = name
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.preprocess = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                            transforms.Lambda(lambda t: t.unsqueeze(0))
                            ])
        self.hook_last_conv(self)

    def hook_last_conv(self, name=None):
        model = self.cnn
        self.hooks = []
        print(type(model))
        if isinstance(model, torchvision.models.GoogLeNet) or self.name:
            self._named_hook(model, self.name if self.name else 'inception5b', '', 0)
        else:
            conv = [None, None]
            conv = self._recursive_hook(model, conv, '', 0)
            self.hooks.append(Hook(conv[0], conv[1])) 
            self.hooks.append(Hook(conv[0], conv[1], backward=True)) 
            print(f'{conv[0]} layer hooked')

    def _named_hook(self, module, target_name, parent_name , depth):
        '''Recursivly search for "target_name" layer in the model and add hook '''
        for name, layer in module.named_children():
            name = parent_name + '_' + name if parent_name else name
            #print('\t'*depth, name)
            if name == target_name:
                self.hooks.append(Hook(name, layer))
                self.hooks.append(Hook(name, layer, backward = True))
                print(f'{name} layer hooked')
            self._named_hook(layer, target_name, name, depth+1)

    def _recursive_hook(self, module, conv, parent_name, depth):
        '''Recursively search for last occuring conv layer in the model and return its name and layer'''
        for name, layer in module.named_children():
            name = parent_name + '_' + name if parent_name else name
            #print('\t'*depth, name)
            if isinstance(layer, nn.Conv2d):
                conv[0], conv[1] = name, layer
            self._recursive_hook(layer, conv, name, depth+1)
        return conv

    def prediction(self, img, index):
        self.cnn.eval()
        pred = cnn(img)
        return pred[0, index].item()

    def get_cam(self, image, index=None):


        img = self.preprocess(image)
        self.cnn.eval()
        pred = self.cnn(img)

        cam = self.hooks[0].output.data
        for index in tqdm(range(cam.shape[1])):
            f_map = cam[:, index, :, :].unsqueeze(1)
            f_map = nn.functional.interpolate(f_map, scale_factor=img.shape[2]//f_map.shape[2], mode='bilinear', align_corners=False)
            f_map -= f_map.min()
            f_map /= f_map.max()
            if index == 0:
                M = img*f_map
            else:
                M = torch.cat((M, img*f_map))
        
        bs = 64
        with torch.no_grad():
            for i in tqdm(range(M.shape[0]//bs)):
                if i == 0:
                    scores = cnn(M[bs*i:bs*(i+1),:,:,:])
                else:
                    scores = torch.cat((scores, cnn(M[bs*i:bs*(i+1),:,:,:])))

        pred = self.cnn(img)
        cam = self.hooks[0].output.data
        while True:
            index = int(input('Enter Index : '))
            tscores = torch.tensor([i - pred[0, index].item() for i in scores[:, index]])
            tscores = self.softmax(tscores)
            tscores = tscores.view(1, -1, 1, 1)

            print(cam.shape)
            out = (self.relu(cam * tscores)).sum(1, keepdim=True)
            print(out.shape)
            out = nn.functional.interpolate(out, scale_factor=img.shape[2] // out.shape[2], mode='bilinear', align_corners=False)
            print(out.shape)
            out = out.squeeze(0).squeeze(0)
            print(out.shape)

            out -= out.min()
            out /= out.max()

            return out

            plt.imshow(out.numpy(), cmap='jet')
            test_img = img.squeeze(0).permute(1, 2, 0).numpy()
            plt.imshow((test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img)), alpha=0.5)
            plt.show()

if __name__ == '__main__':

    cnn = torchvision.models.resnet18(pretrained=True)
    cam = GradCAM(cnn)
    print(os.getcwd())
    image = Image.open(os.getcwd()+'/test/tabbycat_dog.jpg').convert('RGB')
    cam.get_cam(image, 281)