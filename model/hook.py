import torch
import torch.nn as nn
import torchvision


class Hook:

    '''
    Attach forward or backward hook to a layer or module.
    name : name of layer
    layer : nn.Module object
    backward : set True if you want to assign backward pass
    '''
    
    def __init__(
                    self, name, layer,
                    backward=False,
                    forward_input_fn = None,
                    forward_output_fn = None,
                    backward_input_fn = None,
                    backward_output_fn = None
                ):

        
        self.name = name
        self.backward = backward
        self.forward_input_fn = forward_input_fn
        self.forward_output_fn = forward_output_fn
        self.backward_input_fn = backward_input_fn
        self.backward_output_fn = backward_output_fn

        if self.backward:
            print(f'backward hook set on layer {self.name}')
            self.hook = layer.register_backward_hook(self.backward_hook)
        else:
            print(f'forward hook set on layer {self.name}')
            self.hook = layer.register_forward_hook(self.forward_hook)
            
    def forward_hook(self, module, input, output):

        self.input = self.forward_input_fn(input) if self.forward_input_fn else input
        self.output = self.forward_output_fn(output) if self.forward_output_fn else output
        self.module = module
        print(f'forward hook executed on layer {self.name}')
    
    def backward_hook(self, module, input, output):
        self.input = self.backward_input_fn(input) if self.backward_input_fn else input
        self.output = self.backward_output_fn(output) if self.backward_output_fn else output
        self.module = module

        print(f'backward hook executed on layer {self.name}')
        return self.input

    def remove(self):
        self.hook.remove()
        print(f'{"backward" if self.backward else "forward"} hook on layer {self.name} removed')



def set_hook(model, layer_name=None, verbose=False, **kwargs):

    '''
    If layer_name==None serach for last conv layer in the model and assign forward/backward hook
    If layer_name search for a conv layer with name=layer_name in the model and
    assign forward/backward hook
    '''
    hooks = []

    def _named_hook(module, parent_name, depth):

        '''Recursivly search for "target_name" layer in the model and add hook '''
        # For each sub module run the loop
        for name, layer in module.named_children():
            # Construct name of the module
            name = parent_name + '_' + name if parent_name else name
            if verbose : print('\t'*depth, name)
            # if module name is layer_name assign the hook
            if name == layer_name:
                hooks.append(Hook(name, layer, **kwargs))
                hooks.append(Hook(name, layer, backward = True, **kwargs))
                print(f'{name} layer hooked')
            # Recursivly search the module for layer with layer_name
            _named_hook(layer, name, depth+1)

    def _last_layer_hook(module, conv, parent_name, depth):

        '''Recursively search for last occuring conv layer in the model and return its name and layer'''

        # For each sub module run the loop
        for name, layer in module.named_children():
            # Construct name of the layer
            name = parent_name + '_' + name if parent_name else name
            if verbose : print('\t'*depth, name)
            # if it is conv layer save its name and reference in conv list
            if isinstance(layer, nn.Conv2d):
                conv[0], conv[1] = name, layer
            # Recursivly search each module
            _last_layer_hook(layer, conv, name, depth+1)
        # return list with name and reference of last conv layer
        return conv

    if isinstance(model, torchvision.models.GoogLeNet):
        name = 'inception5b' 
    # if name is given, run named_hook
    if layer_name:
        _named_hook(model, '', 0) 
    # if name is not given seach for last conv layer
    else:
        conv = [None, None]
        conv = _last_layer_hook(model, conv, '', 0)
        hooks.append(Hook(conv[0], conv[1], **kwargs)) 
        hooks.append(Hook(conv[0], conv[1], backward=True, **kwargs)) 
        print(f'{conv[0]} layer hooked')

    return hooks
