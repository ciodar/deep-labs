import torch.nn as nn
import torch
import torch.nn.functional as F


class DownsampleA(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        d = self.downsample(x)
        # Here I assume stride is only 2.
        out = torch.concat([d, torch.zeros_like(d)], dim=1)
        return out


class DownsampleB(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.downsample(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False, residual_type='b', stride=1):
        super().__init__()
        self.residual = residual
        self.residual_type = residual_type.lower()
        self.stride = stride
        if residual:
            if self.residual_type == 'a' and (in_channels != out_channels or stride > 1):
                self.project = DownsampleA(in_channels, out_channels, stride)
            if self.residual_type == 'b' and (in_channels != out_channels or stride > 1):
                self.project = DownsampleB(in_channels, out_channels, stride)
            if self.residual_type == 'c':
                self.project = DownsampleB(in_channels, out_channels, stride)
        # vgg has strided conv instead of maxpool
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # first conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # second conv
        out = self.conv2(out)
        out = self.bn2(out)
        # residual connection
        if self.residual:
            res = x if x.shape == out.shape else self.project(x)
            out = out + res
        out = F.relu(out)
        return out


class TinyResNet(nn.Module):
    def __init__(self, layers, num_classes=10, residual=False, residual_type='b', num_channels=16):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(3, num_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        # stack 3 convolutional block of 2n layers each, with
        self.layer1 = self._add_block(num_channels, num_channels, layers[0], self.residual, residual_type, stride=1)
        num_channels *= 2
        self.layer2 = self._add_block(num_channels // 2, num_channels, layers[1], self.residual, residual_type,
                                      stride=2)
        num_channels *= 2
        self.layer3 = self._add_block(num_channels // 2, num_channels, layers[2], self.residual, residual_type,
                                      stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def _add_block(self, in_channels, out_channels, n_blocks, residual=False, residual_type='b', stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, residual, residual_type, stride=stride))
        in_channels = out_channels
        for i in range(1, n_blocks):
            layers.append(ResidualBlock(in_channels, out_channels, residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        # print(out.shape) # 128,16,16,16
        out = self.layer1(out)
        # print(out.shape) # 128,32,8,8
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        # print(out.shape)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
