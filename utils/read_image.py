from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import torch

def alexnet_preprocess():
    return transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def tensor_to_image(t, tensor_out = False):

    t = t.squeeze(dim=0)
    t = t.detach()
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])
    t = t.permute(1, 2, 0)
    mean = torch.reshape(mean, (1, 1, 3))
    std = torch.reshape(std, (1, 1, 3))
    t = torch.clamp(t, min=-50, max=50)
    t = t * std + mean
    t = (t - t.min()) / (t.max() - t.min())
    image = t.numpy()*255
    return t.permute(2, 0, 1) if tensor_out else Image.fromarray(image.astype(np.uint8))

def read_image(path, net='imagenet'):
    image = Image.open(path).convert('RGB')
    if net == 'imagenet' : preprocess = alexnet_preprocess()
    image = preprocess(image).unsqueeze(dim=0)
    return image