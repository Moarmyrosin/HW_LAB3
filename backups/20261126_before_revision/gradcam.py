"""用最佳模型生成正确/错误样例 Grad-CAM，运行前先完成训练。"""
import argparse
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import build,Block
from data import load,testset,MEAN,STD
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--data',default='data'); p.add_argument('--out',default='gradcam.png'); a=p.parse_args()
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False); m=build(c['config']['model']); m.load_state_dict(c['model']); m.eval()
_,_,base=load(a.data,c['config']['seed']); ds=testset(a.data,base)
# SE 门控中的卷积输出为 1×1；解释图需要池化前的空间特征。
blocks=[v for v in m.modules() if isinstance(v,Block)]
layer=blocks[-1] if blocks else [v for v in m.modules() if isinstance(v,torch.nn.Conv2d)][-1]; cache={}
def hook(mod,inp,out): cache['features']=out; out.retain_grad()
h=layer.register_forward_hook(hook); cases=[]; counts=[0,0]
for x,y in ds:
 m.zero_grad(); z=m(x[None]); pred=z.argmax(1).item(); kind=int(pred!=y)
 if counts[kind]>=3: continue
 z[0,pred].backward(); f=cache['features']; weights=f.grad.mean((2,3),keepdim=True); cam=(weights*f).sum(1,keepdim=True).relu(); cam=torch.nn.functional.interpolate(cam,(32,32),mode='bilinear',align_corners=False)[0,0].detach(); cam=cam/(cam.max()+1e-8)
 image=(x*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None]).permute(1,2,0).clamp(0,1); cases.append((image,cam,y,pred)); counts[kind]+=1
 if min(counts)>=3: break
h.remove(); fig,axes=plt.subplots(2,3,figsize=(9,6))
for ax in axes.flat: ax.axis('off')
for ax,(image,cam,y,pred) in zip(axes.flat,cases): ax.imshow(image); ax.imshow(cam,cmap='jet',alpha=.45); ax.set_title(f'True {y}, Pred {pred}')
fig.tight_layout(); fig.savefig(a.out); plt.close(fig)
