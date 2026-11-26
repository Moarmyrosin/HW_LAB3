"""独立数据对象保证验证集不使用随机增强。"""
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import Subset
from pathlib import Path
MEAN=(.4914,.4822,.4465); STD=(.247,.2435,.2616)
def load(root,seed,smoke=False):
 base=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
 aug=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),base])
 train=datasets.CIFAR10(root,train=True,download=True,transform=aug)
 val=datasets.CIFAR10(root,train=True,download=False,transform=base)
 rng=np.random.default_rng(seed); tr=[]; va=[]; y=np.array(train.targets)
 for c in range(10):
  idx=rng.permutation(np.where(y==c)[0]); va.extend(idx[:500]); tr.extend(idx[500:])
 assert not set(tr)&set(va) and len(tr)==45000 and len(va)==5000
 path=Path(root)/f'split_{seed}.npz'
 if path.exists():
  old=np.load(path); assert np.array_equal(old['train'],tr) and np.array_equal(old['val'],va)
 else: np.savez(path,train=tr,val=va)
 if smoke: tr=[i for c in range(10) for i in [j for j in tr if y[j]==c][:8]]; va=[i for c in range(10) for i in [j for j in va if y[j]==c][:4]]
 return Subset(train,tr),Subset(val,va),base
def testset(root,base,smoke=False):
 ds=datasets.CIFAR10(root,train=False,download=True,transform=base)
 return Subset(ds,list(range(32))) if smoke else ds
