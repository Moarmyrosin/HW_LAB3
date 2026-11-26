"""自行实现 CNN、CIFAR ResNet18 与 SE 残差块。"""
import torch
from torch import nn
class SE(nn.Module):
 def __init__(self,c):
  super().__init__(); self.f=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,max(c//16,1),1),nn.ReLU(),nn.Conv2d(max(c//16,1),c,1),nn.Sigmoid())
 def forward(self,x): return x*self.f(x)
class Block(nn.Module):
 def __init__(self,a,b,s=1,se=False):
  super().__init__(); self.f=nn.Sequential(nn.Conv2d(a,b,3,s,1,bias=False),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,1,1,bias=False),nn.BatchNorm2d(b),SE(b) if se else nn.Identity()); self.skip=nn.Identity() if a==b and s==1 else nn.Sequential(nn.Conv2d(a,b,1,s,bias=False),nn.BatchNorm2d(b))
 def forward(self,x): return torch.relu(self.f(x)+self.skip(x))
def build(name):
 if name=='cnn':
  layers=[]; a=3
  for b in [32,64,128]: layers.extend([nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.MaxPool2d(2)]); a=b
  return nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.2),nn.Linear(128,10))
 layers=[nn.Conv2d(3,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU()]; a=64
 for b in [64,128,256,512]:
  layers.extend([Block(a,b,1 if a==b else 2,name=='se'),Block(b,b,se=name=='se')]); a=b
 return nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(512,10))
