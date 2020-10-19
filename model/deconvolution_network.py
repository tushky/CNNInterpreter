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

class DeconvNet(nn.Module):
    '''
    Constructs Deconvolution Network from the pretrained Convolution Network,
    pretrained Network should only have Conv, Relu or MaxPool Layers,
    cnn : pretrained convolutional network
    layer_num : layer number from which you want to perform deconvolution
    guided : set True if you want to perform guided backpropogation
    '''
    def __init__(self, cnn, layer_number, guided=False):

        super().__init__()

        # Put pretrained model into evaluation mode
        self.cnn = cnn
        self.cnn.eval()
        
        self.guided = guided
        self.layer_num = layer_number

        # Construct deconv net from input model
        self.construct_dconvnet()
    
    def construct_dconvnet(self):

        '''
        travel pretrained model layer by layer and construct deconvolution network
        '''

        self.conv_model = OrderedDict()
        self.dconv_model = OrderedDict()

        i = 0
        
        num_conv, num_relu, num_pool = 1, 1, 1

        while i <= self.layer_num:

            layer = self.cnn.features[i]
            
            # If it is a Conv layer, construct Transposed Conv layer for the deconv net
            if isinstance(layer,  nn.Conv2d):

                conv_layer = nn.Conv2d(layer.out_channels,
                                        layer.in_channels,
                                        layer.kernel_size,
                                        layer.stride,
                                        layer.padding
                                        )

                dconv_layer = nn.ConvTranspose2d(layer.out_channels,
                                                layer.in_channels,
                                                layer.kernel_size,
                                                layer.stride,
                                                layer.padding
                                                )

                dconv_layer.weight.data = layer.weight.data
                conv_layer.weight.data = layer.weight.data
                conv_layer.bias.data = layer.bias.data

                self.dconv_model[f'dconv_{num_conv}'] = dconv_layer
                self.conv_model[f'conv_{num_conv}'] = conv_layer
                num_conv += 1

            # If it is a ReLU layer, copy layer for the deconv net
            if isinstance(layer, nn.ReLU):
                self.conv_model[f'relu_{num_relu}'] = layer
                self.dconv_model[f'unrelu_{num_relu}'] = layer
                num_relu += 1

            # If it is a MaxPool layer, set return_indices True so that
            # it returns indices of possitive elements
            if isinstance(layer, nn.MaxPool2d):
                
                maxpool = nn.MaxPool2d(layer.kernel_size,
                                       layer.stride,
                                       layer.padding
                                       )

                maxpool.return_indices = True

                self.conv_model[f'pool_{num_pool}'] = maxpool
                self.dconv_model[f'unpool_{num_pool}'] = nn.MaxUnpool2d(layer.kernel_size,
                                                                        layer.stride,
                                                                        layer.padding
                                                                        )
                num_pool += 1
            i += 1
        
        # Construct forward pass model
        self.conv_model = nn.Sequential(self.conv_model)

        # Construct Deconv Model
        self.dconv_model = nn.Sequential(OrderedDict(reversed(list(self.dconv_model.items()))))
    
    def conv_forward(self, t):
        
        '''
        Performs forward pass on the model till the specified layer number
        '''

        pool_layer = 1
        relu_layer = 1

        # Create dict to remember indices of positive output of MaxPool and ReLU layers during forward pass
        self.pool_indices = {}
        self.relu_indices = {}
        with torch.no_grad():
            for layer in self.conv_model:
                
                # If it is MaxPool layer, remember indices of positive output
                if isinstance(layer, nn.MaxPool2d):
                    t, self.pool_indices[f'unpool_{pool_layer}'] = layer(t)
                    pool_layer += 1

                # If it is ReLU layer, remember indices of positive output
                # Required for Guided Backpropogation only
                elif self.guided and isinstance(layer, nn.ReLU):
                    t = layer(t)
                    self.relu_indices[f'unrelu_{relu_layer}'] = torch.where(t > 0, torch.tensor([1]), torch.tensor([0]))
                    relu_layer += 1

                # If it is Conv layer, perform normal forward pass
                else:
                    t = layer(t)
            
            # return output of the specified layer number
            return t
    
    def dconv_backward(self, t):

        '''
        takes the output of conv model till specified layer
        and pass it through deconv model to project it on input space.
        '''

        pool_layer = 1
        relu_layer = 1
        with torch.no_grad():
            for key, layer in self.dconv_model.named_children():
                
                # if it is MaxPool layer, pass the stored indices of possitive output
                if isinstance(layer, nn.MaxUnpool2d):
                    t = layer(t, self.pool_indices[key])
                    pool_layer += 1
                
                # If it is ReLU layer, use the stored indices of forward pass of conv net
                #  to remove negative values
                elif self.guided and isinstance(layer, nn.ReLU):
                    t = layer(t)
                    t *= self.relu_indices[key]
                    relu_layer += 1
                    #t /= 2.

                # If it is Conv layer, perform normal forward pass on deconv model
                else:
                    t = layer(t)
            return t

    def construct_dconv_input(self, t, num_kernel=9):

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
            #print(index, max_activation[i][0])
            dconv_input[0, max_activation[i][0], index[0], index[1]] = t[0, max_activation[i][0], index[0], index[1]]
            #dconv_input[0, max_activation[i][0], index[0], index[1]] = 1.0
            #print(torch.nonzero(dconv_input))
            yield dconv_input
    
    def get_maps(self, t, num_kernel=1):

        # Obtain output of the conv model
        t = self.conv_forward(t)

        # Construct input for deconv model and obtain its outputs
        for i, dconv_input in tqdm(enumerate(self.construct_dconv_input(t, num_kernel))):
            # pass constructed input to the deconv model
            out = self.dconv_backward(dconv_input)
            # squash deconv output so that visualization is possible
            out = tensor_to_image(out.detach(), True)
            # stack outputs of the decov model
            if i == 0:
                maps = out.unsqueeze(0)
            else:
                maps = torch.cat((maps, out.unsqueeze(0)),dim=0)
        
        return maps
    
    def show_maps(self, image, output_layer, num_kernels = 1, path=False):

        output = self.get_maps(image, num_kernels)
        output_images = torch.stack([process_deconv_output(img.unsqueeze(0)) for img in output], dim=0)
        output_maps  = torchvision.utils.make_grid(output_images, nrow=5, normalize=False)
        plt.gcf().set_size_inches(4*num_kernels, 4)
        plt.axis('off')
        plt.imshow(output_maps.permute(1, 2, 0))
        if path:
            plt.savefig('./data/'+f'clock_deconv_output_layer_{output_layer}', bbox_inches='tight')
            #plt.savefig(path, bbox_inches='tight')
        plt.show()


if __name__ == '__main__' :

    imagenet = models.vgg16(pretrained=True)
    image_path = os.getcwd() + '/test/clock.jpg'
    image = read_image(image_path)
    layer_number=28
    deconvnet = DeconvNet(imagenet, layer_number, guided=True)
    output = deconvnet.show_maps(image, layer_number, 5, path=True)