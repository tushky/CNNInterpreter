import os
import sys
sys.path.insert(1, './utils')
from utils import process_deconv_output, clamp, read_image
import math
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from read_image import tensor_to_image
from torchvision.utils import make_grid, save_image

class DeconvolutionNetwork:

    def __init__(self, model, guided=True):

        if not hasattr(model, 'features'):
            raise ValueError('Make sure all conv layers are in model.features dict')

        self.model = model
        self.guided = guided

        self.construct_deconv_network()
    
    def construct_deconv_network(self):

        self.dconv_model = OrderedDict()
        num_conv, num_relu, num_pool, num_norm = 1, 1, 1, 1


        for name, layer in self.model.features.named_children():

            # If it is a Conv layer, construct Transposed Conv layer for the deconv net
            if isinstance(layer,  nn.Conv2d):

                dconv_layer = nn.ConvTranspose2d(layer.out_channels,
                                                layer.in_channels,
                                                layer.kernel_size,
                                                layer.stride,
                                                layer.padding
                                                )

                dconv_layer.weight.data = layer.weight.data
                layer_name = str(name) + '-' + f'dconv_{num_conv}'
                self.dconv_model[layer_name] = dconv_layer
                num_conv += 1

            # If it is a ReLU layer, copy layer for the deconv net
            elif isinstance(layer, nn.ReLU):
                layer_name = str(name) + '-' + f'unrelu_{num_relu}'
                self.dconv_model[layer_name] = layer
                num_relu += 1

            # If it is a MaxPool layer, set return_indices True so that
            # it returns indices of possitive elements
            elif isinstance(layer, nn.MaxPool2d):
                layer_name = str(name) + '-' + f'unpool_{num_pool}'
                self.dconv_model[layer_name] = nn.MaxUnpool2d(
                                                            layer.kernel_size,
                                                            layer.stride,
                                                            layer.padding
                                                            )
                num_pool += 1
            
            elif isinstance(layer, nn.BatchNorm2d):
                layer_name = str(name) + '-' + f'norm_{num_norm}'
                stats = layer.state_dict()
                #stats['weight'], stats['running_var'] = torch.sqrt(stats['running_var']), stats['weight']**2
                #stats['bias'], stats['running_mean'] = stats['running_mean'], stats['bias']
                self.dconv_model[layer_name] = nn.BatchNorm2d(layer.num_features, affine=True, track_running_stats=True)
                self.dconv_model[layer_name].load_state_dict(stats)
                num_norm += 1

            
            else:
                raise TypeError(f'Network with layer {layer.__class__} is not supported by DecovNet')

        # Construct Deconv Model
        self.deconv_model = nn.Sequential(OrderedDict(reversed(list(self.dconv_model.items()))))
        print(self.deconv_model)
    
    def conv_forward(self, t, target_layer):
        
        '''
        Performs forward pass on the model till the specified layer number
        '''

        pool_layer = 1
        relu_layer = 1
        norm_layer = 1

        # Create dict to remember indices of positive output of MaxPool and ReLU layers during forward pass
        self.pool_indices = {}
        self.relu_indices = {}

        with torch.no_grad():
            for name, layer in self.model.features.named_children():

                print(name)
                # If it is MaxPool layer, remember indices of positive output
                if isinstance(layer, nn.MaxPool2d):
                    layer.return_indices = True
                    t, self.pool_indices[f'unpool_{pool_layer}'] = layer(t)
                    layer.return_indices = False
                    pool_layer += 1
                
                #if isinstance(layer, nn.BatchNorm2d):

                # If it is ReLU layer, remember indices of positive output
                # Required for Guided Backpropogation only
                elif self.guided and isinstance(layer, nn.ReLU):
                    t = layer(t)
                    self.relu_indices[f'unrelu_{relu_layer}'] = torch.where(t > 0, torch.tensor([1]), torch.tensor([0]))
                    relu_layer += 1

                # If it is Conv layer, perform normal forward pass
                else:
                    t = layer(t)
                print(f'output after layer {name} is {t.max()}')
                if str(name) == str(target_layer):
                    print(f'layer {name} found with target layer {target_layer}')
                    print(self.pool_indices.keys(), self.relu_indices.keys())
                    break
            
        # return output of the specified layer number
        return t
    
    def deconv_backward(self, t, target_layer):

        '''
        takes the output of conv model till specified layer
        and pass it through deconv model to project it on input space.
        '''

        pool_layer = 1
        relu_layer = 1

        found_target_layer = False
        self.deconv_model.eval()
        with torch.no_grad():
            for name, layer in self.deconv_model.named_children():

                conv_layer_name, dconv_layer_name = name.split('-')

                if conv_layer_name == str(target_layer):
                    found_target_layer = True
                
                if conv_layer_name != str(target_layer) and not found_target_layer:
                    continue
                print(name)
                # if it is MaxPool layer, pass the stored indices of possitive output
                if isinstance(layer, nn.MaxUnpool2d):
                    t = layer(t, self.pool_indices[dconv_layer_name])
                    pool_layer += 1
                
                # If it is ReLU layer, use the stored indices of forward pass of conv net
                #  to remove negative values
                elif self.guided and isinstance(layer, nn.ReLU):
                    t = layer(t)
                    t *= self.relu_indices[dconv_layer_name]
                    relu_layer += 1
                
                elif isinstance(layer, nn.BatchNorm2d):
                    stats = layer.state_dict()
                    t = ((t - stats['bias'].view(1, -1, 1, 1)) *\
                         (torch.sqrt(stats['running_var'].view(1, -1, 1, 1)+1e-05))/\
                             (stats['weight'].view(1, -1, 1, 1)+1e-05)) +\
                                  stats['running_mean'].view(1, -1, 1, 1)

                # If it is Conv layer, perform normal forward pass on deconv model
                else:
                    t = layer(t)
                print(f'output after layer {name} is {t.max()}')
        return t

    def construct_deconv_input(self, t, num_kernel=9):

        '''
        Takes the output of the conv model and prepares it for the deconv model.
        t : output tensor of the conv model
        num_kernel : number of kernels you want to visualize
        '''

        assert len(t.shape) == 4, f"Incorrect input shape, expected tensor of shape 4, recived tensor of shape {t.shape}"
        max_activation = [None] * t.shape[1]

        # For each of the feature map find the maximum activation
        for kernal in range(t.shape[1]):
            max_activation[kernal] = (kernal, torch.max(t[0, kernal, :, :]).item(),
             torch.argmax(t[0, kernal, :, :]).item())

        # Sort features according to their maximum activation value    
        max_activation.sort(key=lambda x: x[1], reverse=True)

        # Construct "num_kernel" number of input tensor for deconv model
        for i in range(num_kernel):
            dconv_input = torch.zeros_like(t)
            index = np.unravel_index(max_activation[i][2], (t.shape[2], t.shape[3]))
            dconv_input[0, max_activation[i][0], index[0], index[1]] = t[0, max_activation[i][0], index[0], index[1]]
            yield dconv_input
    
    def get_maps(self, t, target_layer, num_kernel=1):

        # Obtain output of the conv model
        t = self.conv_forward(t, target_layer)

        # Construct input for deconv model and obtain its outputs
        for i, dconv_input in tqdm(enumerate(self.construct_deconv_input(t, num_kernel))):
            # pass constructed input to the deconv model
            out = self.deconv_backward(dconv_input, target_layer)
            # squash deconv output so that visualization is possible
            out = tensor_to_image(out.detach(), True)
            # stack outputs of the decov model
            if i == 0:
                maps = out.unsqueeze(0)
            else:
                maps = torch.cat((maps, out.unsqueeze(0)),dim=0)
        
        return maps
    
    def show_maps(self, image, target_layer, num_kernels = 1, path=False):

        output = self.get_maps(image, target_layer, num_kernels)
        output_images = torch.stack([process_deconv_output(img.unsqueeze(0)) for img in output], dim=0)
        output_maps  = torchvision.utils.make_grid(output_images, nrow=5, normalize=False)
        plt.gcf().set_size_inches(4*num_kernels, 4)
        plt.axis('off')
        plt.imshow(output_maps.permute(1, 2, 0))
        if path:
            plt.savefig('./data/'+f'clock_deconv_output_layer_{target_layer}', bbox_inches='tight')
            #plt.savefig(path, bbox_inches='tight')
        plt.show()
            

if __name__ == '__main__':

    model = models.vgg16(pretrained=True)
    image_path = os.getcwd() + '/test/cat_dog.png'
    image = read_image(image_path)
    target_layer=8
    deconvnet = DeconvolutionNetwork(model, guided=True)
    deconvnet.show_maps(image, target_layer, 1, path=True)
    deconvnet.show_maps(image, 15, 5, path=True)
    deconvnet.show_maps(image, 20, 5, path=True)
    deconvnet.show_maps(image, 25, 5, path=True)
    deconvnet.show_maps(image, 28, 5, path=True)
