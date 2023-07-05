import torch.nn as nn
import torch 
import torch.nn.functional as F

# A very simple CNN model.
class BaseCNN(nn.Module):
    def __init__(self,
                 activation: str = 'ReLU'
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.activation = getattr(nn, activation)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# A more complex CNN model.
class CNN(nn.Module):
    def __init__(self,
                num_classes: int = 10,
                activation: str = 'ReLU',
                num_conv_layers: int = 2,
                num_fc_layers: int = 2,
                base_out_filters: int = 32,
                image_size: int = 32
                ):
        super().__init__()
        conv_layers,fc_layers = [],[]
        self.activation = getattr(nn, activation)
        in_channels = 3
        out_channels = base_out_filters
        for i in range(num_conv_layers):
            out_channels = base_out_filters * 2**i
            layer = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            in_channels = out_channels
            bn = nn.BatchNorm2d(out_channels)
            conv_layers.append(layer)
            conv_layers.append(nn.AvgPool2d(2))
            conv_layers.append(bn)
            conv_layers.append(self.activation())
        self.conv_layers = nn.Sequential(*conv_layers)
        in_features = (image_size // 2**num_conv_layers)**2 * out_channels
        if num_fc_layers == 0:
            out_features = in_features
        for i in range(num_fc_layers):
            out_features = in_features // 2**(i+1)
            layer = nn.Linear(in_features, out_features)
            in_features = out_features
            fc_layers.append(layer)
            bn = nn.BatchNorm1d(out_features)
            fc_layers.append(bn)
            fc_layers.append(self.activation())
        fc_layers.append(nn.Linear(out_features, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)
        

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out