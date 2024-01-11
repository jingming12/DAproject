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
        mask = torch.where(torch.rand_like(output) < 0.2, 0.0, 1.0) #random mask
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
        stored_outputs_hook = []
        alternate = 0
        for module in self.resnet.modules():
            if isinstance(module, nn.Conv2d):
                alternate += 1
                if alternate % 3 == 0:
                    stored_outputs_hook.append(module.register_forward_hook(activation_shaping_hook))

    def forward(self, x):
        return self.resnet(x)
######################################################
