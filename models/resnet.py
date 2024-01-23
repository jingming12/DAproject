import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)

######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
#class ActivationShapingModule(nn.Module):
#...
#
# OR as a function that shall be hooked via 'register_forward_hook'
def activation_shaping_hook(module, input, output):
        # set the mask & get A_bin;M_bin
        mask = torch.where(torch.rand_like(output) < 0.4, 0.0, 1.0) #random mask
        A_bin = torch.where(output <= 0, torch.tensor(0.0), torch.tensor(1.0))
        M_bin = torch.where(mask <= 0, torch.tensor(0.0), torch.tensor(1.0))
        # return the element-wise product of activation map and mask
        return A_bin * M_bin
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.stored_outputs_hook = []
        self.activation_maps = {}
        alternate = 0
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d): ## can be changed
                alternate += 1
                if alternate % 3 == 0:
                    hook = module.register_forward_hook(self.create_hook(name))
                    self.stored_outputs_hook.append(hook)
    def create_hook(self, name):
        def hook(module, input, output):
            shaped_output = activation_shaping_hook(module, input, output)
            self.activation_maps[name] = shaped_output
        return hook
    def forward(self, x, target_activation_map_name=None):
        x = self.resnet(x)
        if target_activation_map_name and target_activation_map_name in self.activation_maps:
            shaped_activation_map = self.activation_maps[target_activation_map_name]
            x = x * shaped_activation_map
        return x
    def get_activation_maps(self, target_x, layer_name):
        with torch.no_grad():
            self(target_x)
        return self.activation_maps.get(layer_name)
######################################################
