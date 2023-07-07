import torch.nn as nn
import torch 
import torch.nn.functional as F

import numpy as np

class Discriminator(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(Discriminator, self).__init__()
        self.backbone = backbone
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.backbone(x)).view(-1)

class Generator(nn.Module):
    def __init__(self,
                backbone: nn.Module, 
                criterion,
                adapter_type: str='conv',
                ):
        super(Generator, self).__init__()
        self.backbone = backbone
        self.adapter = Adapter(
            adapter_type=adapter_type,
        )
        # generation hyperparameters
        self.criterion = criterion

    def forward(self, x):
        return self.backbone(x)

    def generate(self, x, y ):
        x.requires_grad = True
        out = self.backbone(x)
        errG_cls = self.criterion(out, y)
        errG_cls.backward(retain_graph=True)
        preds_cls = torch.argmax(out, dim=1)
        J = x.grad.detach()
        J_prime = self.adapter(J)
        return J_prime, (J, errG_cls, preds_cls)
        

class Adapter(nn.Module):
    def __init__(self, 
                adapter_type: str='conv',
                *args, **kwargs):
        super().__init__(*args, 
                         **kwargs)
        if adapter_type == 'conv':
            self.adapter = nn.Conv2d(3, 3, 1)
            self.activation = nn.Tanh()
        elif adapter_type == 'conv-noactivation':
            self.adapter = nn.Conv2d(3, 3, 1)
            self.activation = None
        elif adapter_type == 'normalize':
            self.adapter = None
            self.activation = None
        self.normalization = F.layer_norm

        # initialize weights
        if self.adapter is not None:
            self._initialize_weights()
    
    def _initialize_weights(self):
        n = self.adapter.kernel_size[0] * self.adapter.kernel_size[1] * self.adapter.out_channels
        nn.init.normal_(self.adapter.weight.data, 0.0, np.sqrt(2.0 / n))
        nn.init.constant_(self.adapter.bias.data, 0.0)

    def forward(self, x):
        N, C, H, W = x.shape
        out = self.normalization(x,[C,H,W])
        if self.adapter is not None:
            out = self.adapter(out)
        if self.activation is not None:
            out = self.activation(out)
        return out
