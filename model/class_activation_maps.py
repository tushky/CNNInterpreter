"""
Created by : Tushar Gadhiya
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import relu, softmax

import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

from hook import set_hook
from classes import classes as class_labels
from utils import postprocess, read_image, process_deconv_output
from deconvolution_network import DeconvolutionNetwork as DeconvNet


class ClassActivationMaps:

    """
        Generate Class Activation Maps for the input image using one of the following methods.
            - Class Activation Maps (CAM)
            - Gradient weighted Class Activation Maps (Grad-CAM)
            - Gradient weighted Class Activation Maps++ (Grad-CAM++)
            - Score weighted Class Activation Maps (Score-CAM)

    Args:
        model (nn.Module): any pretrained convolutional neural network.
        layer_name (str or None): name of the conv layer you want to visualize.
            If None, the last occuring conv layer in the model will be used.

    Attributes:
        model (nn.Module): the pretrained network
        layer_name(str or none): name of the conv layer
        hooks (list): contains handles for forward and backward hooks
        interractive (bool): determines wether to remove the hooks after obtaining cam.
        methods (list): list of acceptable methods

    Example:
        model = torchvision.models.resnet34(pretrained=True)
        image = read_image('test.img')
        cam = ClassActivationMaps(model)
        cam.show_cam(image, method='gradcam++')
    """

    def __init__(self, model, layer_name=None):
        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.hooks = None
        self.interactive = False
        self.methods = ['cam', 'gradcam', 'gradcam++', 'scorecam']


    def cam(self, tensor, class_index):

        """
            Implimentation of vanilla Class Activation Map. Works on specific type
            of CNN network only. Last two layer of the network must have to be
            globalavgpool layer followed by single fully connected layer. For example
            it will not work on VGG16 or AlexNet. but it will work on ResNet or GoogleNet.

        Args:
            tensor (torch.tensor): input image tensor.
            class_index (int or None): index of class for which class activation map
                will be generated. If None the class with the largest logit value will
                be used.
        """
        # obtain prediction for the given image
        with torch.no_grad():
            pred = self.model(tensor)
        print(f'predicted class : {pred.argmax().item()}')

        # if index is not provided use the class index with the largest logit value
        if not class_index:
            class_index = pred.argmax().item()

        class_activation_map = self.hooks[0].output.data.mean(dim=0, keepdim=True)
        class_activation_map = class_activation_map.permute(0, 2, 3, 1)

        weight = list(self.model.named_children())[-1][1].weight.data

        weight = weight.t()
        class_activation_map = class_activation_map @ weight[:, [class_index]]
        class_activation_map = class_activation_map.permute(0, 3, 1, 2)

        return class_activation_map


    def gradcam(self, tensor, class_index):

        """
            Implimentation of gradient weighted class activation maps (Grad-CAM).
            It generalizes vanilla class activation maps and removes the limitation
            on the structure of the network.
        Args:
            tensor (torch.tensor): input image tensor.
            class_index (int or None): index of class for which class activation map
                will be generated. If None the class with the largest logit value will
                be used.
        """

        # obtain prediction for the given image
        pred = self.model(tensor)
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
        loss = torch.sum(pred * target)
        # remove previous gradients
        self.model.zero_grad()
        # obtain gradients
        loss.backward(retain_graph=True)
        # obtain graients for the last conv layer from the backward_hook
        grad = self.hooks[1].output[0].data
        # obtain weights for each feature map of last conv layer using gradients of that layer
        grad = torch.mean(grad, (0, 2, 3), keepdim=True)
        # obtain output of last conv layer from the forward_hook
        conv_out = self.hooks[0].output.data.mean(dim=0, keepdim=True)
        # obtain weighed feature maps, keep possitive influence only
        class_activation_map = relu((conv_out * grad).sum(1, keepdim=True))

        return class_activation_map


    def gradcamplus(self, tensor, class_index):

        """
            Implimentation of gradient weighted class activation maps++ (Grad-CAM++).
            It generalizes Grad-CAM and by extention CAM. It produces better visualization
            by considering pixels.
        Args:
            tensor (torch.tensor): input image tensor.
            class_index (int or None): index of class for which class activation map
                will be generated. If None the class with the largest logit value will
                be used.
        """
        a = torch.Tensor([1.1])
        pred = self.model(tensor)
        print(f'predicted class : {pred.argmax().item()}')
        target = torch.zeros_like(pred)
        # if index is not provided use the class index with the largest logit value
        if not class_index:
            class_index = pred.argmax().item()
        print(f'showing cam for class : {class_index} {class_labels[class_index]}')
        target[0, class_index] = 1
        # allow gradients for target vector
        target.require_grad=True
        # obtain loss for the class "class_index"
        #loss = torch.exp(torch.sum(pred*target))
        loss = a ** torch.sum(pred*target)
        # remove previous gradients
        self.model.zero_grad()
        # obtain gradients
        loss.backward(retain_graph=True)
        # obtain graients for the last conv layer from the backward_hook
        # grad = dY_c/dA
        # First order derivative of score of class 'c' with respect to output of last conv layer.
        grad = self.hooks[1].output[0].data.mean(dim=0, keepdim=True)
        # Second order derivative of score of class 'c' with respect to output of last conv layer.
        # Since secod order derivative of relu layer is zero,
        # the formula is simplified to just square of the first order derivative.
        grad_2 = (grad ** 2)
        # Third order derivative of score of class 'c' with respect to output of last conv layer.
        # Since secod and third order derivative of relu layer is zero,
        # the formula is simplified to just cube of the first order derivative.
        grad_3 = (grad ** 3) * torch.log(a)
        #grad *= loss.item()
        # get global average of gradients of each feature map
        grad_3_sum = torch.mean(grad, (2, 3), keepdim=True)
        # prepare for alpha denominator
        grad_3 = grad_3 * grad_3_sum
        # get alpha
        alpha_d = 2 * grad_2 + grad_3
        alpha_d = torch.where(alpha_d != 0.0, alpha_d, torch.Tensor([1.0]))
        alpha = torch.div(grad_2, alpha_d+1e-06)
        alpha_t = torch.where(relu(grad)>0, alpha, torch.Tensor([0.0]))
        alpha_c = torch.sum(alpha_t, dim=(2, 3), keepdim=True)
        alpha_cn = torch.where(alpha_c !=0, alpha_c, torch.Tensor([1.0]))
        alpha /= alpha_cn
        # get final weights of each feature map
        weight = (alpha * relu(grad)).sum((2, 3), keepdim=True)
        # obtain output of last conv layer from the forward_hook
        conv_out = self.hooks[0].output.data.mean(dim=0, keepdim=True)
        # obtain weighed feature maps, keep possitive influence only
        class_activation_map = relu(conv_out * weight).sum(1, keepdim=True)

        return class_activation_map


    def scorecam(self, tensor, class_index, batch_size=64):

        """
            Implimentation of score weighted class activation map (Score-CAM).
            Unlike gradient based CAM (CAM, Grad-CAM, Grad-CAM++), Score-CAM does
            not uses gradients.
        Args:
            tensor (torch.tensor): input image tensor.
            class_index (int or None): index of class for which class activation map
                will be generated. If None the class with the largest logit value will
                be used.
            batch_size = batch size used for calculating scores of the masked inputs.
        """
        assert (tensor.shape[0]==1), f'Invalid input shape, batch dim should be 1 for scorecam'

        # obtain prediction of network
        pred = self.model(tensor)

        if not class_index:
            class_index = pred.argmax().item()

        # obtain output of last conv layer from forward_hook
        class_activation_map = self.hooks[0].output.data

        masked_inputs = []
        # mask input image using each feature map of last conv layer
        for index in tqdm(range(class_activation_map.shape[1])):

            # extract one feature map
            f_map = class_activation_map[:, index, :, :].unsqueeze(1)

            # resize it to match input image size
            f_map = nn.functional.interpolate(f_map,
                                                size=(tensor.shape[2], tensor.shape[3]),
                                                mode='bilinear',
                                                align_corners=False
                                            )
            # Normalize in range [0, 1]
            f_map -= f_map.min()
            f_map /= (f_map.max() + 1e-05)

            # obtain masked input image using resized and normalized feature map
            masked_inputs.append(tensor * f_map)

        masked_inputs = torch.cat(masked_inputs, dim=0)

        dataloader = DataLoader(masked_inputs, batch_size=batch_size)
        # obtain prediction of model for each masked input image
        with torch.no_grad():

            scores = []

            for t in dataloader:
                scores.append(self.model(t))
   
        scores = torch.cat(scores, dim=0)
        # for each masked input image calculate the increase in logit value
        tscores = torch.Tensor([i - pred[0, class_index].item() for i in scores[:, class_index]])
        # Normalize so that they sum to 1.
        tscores = softmax(tscores, dim=0)
        # reshape to the size of [1, C, 1, 1]
        tscores = tscores.view(1, -1, 1, 1)
        # obtained weighted feature map using normalized scores
        class_activation_map = (relu(class_activation_map * tscores)).sum(1, keepdim=True)
        return class_activation_map


    def get_guided_cam(self, tensor, method, class_index=None, guided=True):

        """
            combines class activation based methods with deconvolution network
            based methods
        """

        class_activation_map = self.get_cam(tensor, method, class_index)
        deconvnet = DeconvNet(self.model, guided)
        maps = deconvnet.get_maps(tensor, self.layer_name, 1)

        output_images = torch.stack(
            [process_deconv_output(img.unsqueeze(0)) * class_activation_map.unsqueeze(0)\
                 for img in maps], dim=0)

        output_maps  = torchvision.utils.make_grid(output_images, nrow=5, normalize=False)
        plt.gcf().set_size_inches(4, 4)
        plt.axis('off')
        plt.imshow(output_maps.permute(1, 2, 0))
        plt.show()


    def get_cam(self, tensor, method, class_index=None):

        """
            return class_activation_map generated by specified method
        """

        if method not in self.methods:
            raise ValueError(f'invalid method name {method},\
                 plese choose one of the following method: cam, gradcam, gradcam++, scorecam')

        if not self.hooks:
            self.hooks = set_hook(self.model, self.layer_name)

        if method == 'cam' :
            cam_map = self.cam(tensor, class_index)
        elif method == 'gradcam' :
            cam_map = self.gradcam(tensor, class_index)
        elif method == 'gradcam++':
            cam_map = self.gradcamplus(tensor, class_index)
        elif method == 'scorecam':
            cam_map = self.scorecam(tensor, class_index)
        else:
            raise ValueError(f'Invalid method name {method}')

        # Remove hooks from last layer just in case if you train your model after obtaining CAM
        if not self.interactive:
            for hook in self.hooks:
                hook.remove()
            self.hooks = None
        # resize it to the shape of input image
        cam_map = nn.functional.interpolate(cam_map,
                                            size=(tensor.shape[2], tensor.shape[3]),
                                            mode='bilinear',
                                            align_corners=False
                                            )
        # remove batch and channel dimention
        cam_map = cam_map.squeeze(0).squeeze(0)

        # Normalize
        cam_map -= cam_map.min()
        cam_map /= (cam_map.max() + 1e-05)

        return cam_map


    def show_cam(self, tensor, class_index=None, method='gradcam', path=None):

        """
            display class_activation_map generated by specified method
        """
        # specify size of the image
        plt.gcf().set_size_inches(8, 8)
        plt.axis('off')
        # plot CAM
        class_activation_map = self.get_cam(tensor, method, class_index)
        img = postprocess(tensor.mean(dim=0, keepdim=True))
        # plot input image
        plt.imshow(class_activation_map, cmap='jet')
        plt.imshow(img, alpha=0.5)
        # save overlapping output
        if path:
            try:
                plt.savefig(path, bbox_inches='tight')
            except:
                print(f'invalid path {path}')
        #plt.show()
        plt.cla()

if __name__ == '__main__':
    from utils import noisy_inputs
    cnn = torchvision.models.googlenet(pretrained=True)
    #cnn = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
    FILE = 'cat_dog.png'
    image = read_image(os.getcwd() +'/images/'+ FILE)
    images = noisy_inputs(image, std=0.1, num_imgs=50)
    cam = ClassActivationMaps(cnn)
    cam.show_cam(image, method='gradcam++', class_index=243, path='noise')
        