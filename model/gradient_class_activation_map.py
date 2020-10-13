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
from read_image import read_image, tensor_to_image
from sign_classification import SignNet
import pickle

class Hook:
    
    def __init__(self, name, layer, backward=False):
        
        self.name = name
        self.backward = backward
        if self.backward:
            print(f'backward hook set on layer {self.name}')
            self.handle = layer.register_backward_hook(self.hook_fn)
        else:
            print(f'forward hook set on layer {self.name}')
            self.handle = layer.register_forward_hook(self.hook_fn)
            
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        self.module = module
        print(f'{"backward" if self.backward else "forward"} hook executed on layer {self.name}')

    def remove(self):
        self.handle.remove()
        print(f'{"backward" if self.backward else "forward"} hook on layer {self.name} removed')


class GradCAM(nn.Module):

    #@profile
    def __init__(self, cnn, name=None):
        
        super().__init__()
        self.cnn = cnn
        self.name = name
        self.relu = nn.ReLU()
        self.preprocess = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                            transforms.Lambda(lambda t: t.unsqueeze(0))
                            ])
        self.hook_last_conv(self)
        self.hooks = SetHook(cnn, name)

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

    def get_cam(self, img, index=None):
        
        self.cnn.eval()
        pred = self.cnn(img)
        print(f'predicted class : {pred.argmax().item()}')
        target = torch.zeros_like(pred)
        if index:
            target[0, index] = 1
        else:
            target[0, pred.argmax().item()] = 1
        target.require_grad=True

        loss = torch.sum(pred*target)
        print(loss)

        self.cnn.zero_grad()
        loss.backward(retain_graph=True)

        grad = self.hooks[1].output[0].data
        grad = torch.mean(grad, (2,3), keepdim=True)

        cam = self.hooks[0].output.data
        out = self.relu((cam * grad).sum(1, keepdim=True))
        out = nn.functional.interpolate(out, scale_factor=img.shape[2] // out.shape[2], mode='bilinear', align_corners=False)
        out = out.squeeze(0).squeeze(0)
        out -= out.min()
        out /= out.max()

        return out

    def show_cam(self, img, index=None):

            f = plt.figure()
            cam = self.get_cam(img, index)
            print(cam.shape)
            f.add_subplot(1, 2, 1)
            plt.imshow(cam, cmap='jet')
            test_img = img.squeeze(0).permute(1, 2, 0).numpy()
            print(test_img.shape)
            f.add_subplot(1, 2, 2)
            #plt.imshow(test_img)
            plt.imshow((test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img)))
            plt.show()
    

if __name__ == '__main__':

    cnn = SignNet()
    cnn.load_state_dict(torch.load('trained_model', map_location=torch.device('cpu')))
    cam = GradCAM(cnn)
    #image = read_image(os.getcwd()+'/test/spider.png')
    #image = read_image('../test/28.png')
    test_data = pickle.load(open("test.p", "rb"))
    x_test, y_test = test_data["features"].astype('float32'), test_data["labels"]
    x_test = torch.from_numpy(x_test).permute(0, 3, 1, 2)
    mean = torch.tensor([73.7285, 67.2971, 69.7296]).view(1, -1, 1, 1)
    std = torch.tensor([70.2567, 67.4864, 69.1561]).view(1, -1, 1, 1)
    x_test = (x_test - mean) / std
    index = 467
    cam.show_cam(x_test[index].unsqueeze(0))
    print(f'actual class {y_test[index]}')