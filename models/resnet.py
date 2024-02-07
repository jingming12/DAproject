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
#def activation_shaping_hook(module, input, output):
        # set the mask & get A_bin;M_bin
        #mask = torch.where(torch.rand_like(output) < 0, 0.0, 1.0) #random mask
        #A_bin = torch.where(output <= 0, torch.tensor(0.0), torch.tensor(1.0))
        #M_bin = torch.where(mask <= 0, torch.tensor(0.0), torch.tensor(1.0))
        # return the element-wise product of activation map and mask
        #return A_bin * M_bin
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
class ASHResNet18(nn.Module):
    def __init__(self, activation_interval=3, layer_types=nn.Conv2d):  # modify here~
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.stored_outputs_hook = []
        self.activation_maps = {}
        self.create_and_register_hooks(activation_interval, layer_types)
    def create_and_register_hooks(self, activation_interval, layer_types):
        counter = 0
        for name, module in self.resnet.named_modules():
            if isinstance(module, layer_types):
                if counter % activation_interval == 0:
                    self.create_hook(name, module)
                counter += 1
    def create_hook(self, name, module):
        #print("Registering hook on:", name)
        hook = module.register_forward_hook(lambda mod, inp, out: self.store_activation_map(name, mod, inp, out))
        self.stored_outputs_hook.append(hook)
    def store_activation_map(self, name, module, input, output):
        shaped_output = self.activation_shaping_hook(module, input, output)
        self.activation_maps[name] = shaped_output
       # print(f"Storing activation map for layer: {name}")

    def activation_shaping_hook(self, module, input, output):
        mask = torch.where(torch.rand_like(output) < 0.05, 0.0, 1.0)  # 0.05,0.2,0.6
        A_bin = torch.where(output <= 0, torch.tensor(0.0), torch.tensor(1.0))
        M_bin = torch.where(mask <= 0, torch.tensor(0.0), torch.tensor(1.0))
        shaped_output = A_bin * M_bin
       # print(f"Generating activation map for layer: {module}, output shape: {output.shape}")
        return shaped_output

    def forward(self, x, activation_maps=None):
        x = self.resnet.conv1(x)
        if activation_maps and 'conv1' in activation_maps:
            x = x * activation_maps['conv1']
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        for layer_name, layer in [('layer1', self.resnet.layer1),
                                  ('layer2', self.resnet.layer2),
                                  ('layer3', self.resnet.layer3),
                                  ('layer4', self.resnet.layer4)]:
            for sublayer_name, module in layer.named_children():
                full_name = f"{layer_name}.{sublayer_name}"
                x = module(x)
                if activation_maps and full_name in activation_maps:
                    x = x * activation_maps[full_name]
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

    def get_activation_maps(self, target_x):
        self.activation_maps.clear()
        with torch.no_grad():
            self(target_x)
        return self.activation_maps
######################################################
