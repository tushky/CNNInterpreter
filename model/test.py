'''
Created By : Tushar Gadhiya
'''

import os
import torchvision
from utils import read_image
import torch
from class_activation_maps import ClassActivationMaps
from deconvolution_network import DeconvolutionNetwork
from sensitivity_maps import SensitivityMaps

'''
#cnn = torchvision.models.googlenet(pretrained=True)
cnn = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
model = 'shufflenet'
cam = ClassActivationMaps(cnn)
for method in ['gradcam']:
    for image_name in os.listdir('./images'):
        image = read_image(os.getcwd() +'/images/' + image_name)
        f_n = image_name.split('.')[0]
        save_path = os.getcwd() +'/results/' + f_n + '_' + method + '_' + model
        cam.show_cam(image, method=method , path=save_path)

cnn = torchvision.models.vgg16(pretrained=True)
#cnn = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
model = 'vgg16'
saliency = SensitivityMaps(cnn)
for method in ['smooth']:
    for image_name in os.listdir('./images'):
        image = read_image(os.getcwd() +'/images/'+ image_name)
        f_n = image_name.split('.')[0]
        save_path = os.getcwd() +'/results/' + f_n + '_' + method + '_' + model
        saliency.show_map(image, method=method, path=save_path)

cnn = torchvision.models.vgg16(pretrained=True)
print(cnn)

#cnn = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
model = 'vgg16'
for guided in [True]:
    deconv = DeconvolutionNetwork(cnn, guided=guided)
    for image_name in os.listdir('./images'):
        image = read_image(os.getcwd() +'/images/'+ image_name)
        f_n = image_name.split('.')[0]
        save_path = os.getcwd() +'/results/' + f_n + '_' + str(guided) + '_' + model
        deconv.show_maps(image, 28, num_kernels=1, path=save_path)


from utils import postprocess
import matplotlib.pyplot as plt
for image_name in os.listdir('./images'):
    image = read_image(os.getcwd() +'/images/'+ image_name)
    f_n = image_name.split('.')[0]
    save_path = os.getcwd() +'/results/' + f_n
    image = postprocess(image)
    plt.gcf().set_size_inches(8, 8)
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(save_path, bbox_inches='tight')
    plt.cla()
'''