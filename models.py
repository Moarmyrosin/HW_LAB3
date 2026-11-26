"""自行实现基础 CNN、CIFAR ResNet18 和 SE；未加载任何预训练权重。"""
import torch
from torch import nn


class SE(nn.Module):
    """全局平均池化压缩空间维，学习逐通道门控权重。"""
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 16, 1)
        self.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                               nn.ReLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x)


class Block(nn.Module):
    """两个 3×3 卷积的基础残差块；尺寸改变时使用 1×1 投影捷径。"""
    def __init__(self, a, b, s=1, se=False):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(a, b, 3, s, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(),
            nn.Conv2d(b, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b),
            SE(b) if se else nn.Identity())
        self.skip = (nn.Identity() if a == b and s == 1 else
                     nn.Sequential(nn.Conv2d(a, b, 1, s, bias=False), nn.BatchNorm2d(b)))

    def forward(self, x):
        return torch.relu(self.f(x) + self.skip(x))


def build(name, num_classes=10):
    if name == 'cnn':
        layers, channels = [], 3
        for width in [32, 64, 128]:
            layers.extend([nn.Conv2d(channels, width, 3, padding=1),
                           nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)])
            channels = width
        return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                             nn.Dropout(.2), nn.Linear(128, num_classes))
    if name not in ['resnet18', 'se']:
        raise ValueError(f'未知模型：{name}')
    # 适配 32×32：3×3 stem、stride=1，取消 ImageNet 模型最初的 maxpool。
    layers = [nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU()]
    channels = 64
    for width in [64, 128, 256, 512]:
        layers.extend([Block(channels, width, 1 if channels == width else 2, name == 'se'),
                       Block(width, width, se=name == 'se')])
        channels = width
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(512, num_classes))
