"""训练、恢复与独立测试；smoke 结果不可作为正式性能。"""
import argparse,json,random,time,csv,platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
from models import build
from data import load,testset,MEAN,STD
p=argparse.ArgumentParser()
for name,default,typ in [('model','cnn',str),('epochs',30,int),('batch_size',128,int),('seed',42,int),('data','data',str),('out','runs/cnn',str),('resume','',str),('device','auto',str),('patience',0,int)]: p.add_argument('--'+name,default=default,type=typ)
p.add_argument('--smoke',action='store_true'); p.add_argument('--test',action='store_true'); p.add_argument('--amp',action='store_true')
a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
if not a.resume and not a.test and (out/'last.pt').exists(): raise RuntimeError('输出目录已有 checkpoint，请使用 --resume 或新的 --out')
if a.model not in ['cnn','resnet18','se']: raise ValueError('未知模型')
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
torch.set_num_threads(4)
device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
m=build(a.model).to(device); opt=torch.optim.SGD(m.parameters(),lr=.1,momentum=.9,weight_decay=5e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs); scaler=torch.amp.GradScaler('cuda',enabled=a.amp and device=='cuda')
tr,va,base=load(a.data,a.seed,a.smoke)
def loader(ds,shuffle=False): return DataLoader(ds,batch_size=a.batch_size,shuffle=shuffle,num_workers=0)
def step(ds,train=False):
 m.train(train); loss_sum=correct=n=0; ys=[]; ps=[]; probs=[]; ims=[]
 with torch.set_grad_enabled(train):
  for x,y in loader(ds,train):
   x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=device,enabled=a.amp and device=='cuda'): z=m(x); loss=nn.functional.cross_entropy(z,y)
   if train: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
   loss_sum+=loss.item()*len(y); correct+=(z.argmax(1)==y).sum().item(); n+=len(y)
   if not train: ys.extend(y.cpu().tolist()); ps.extend(z.argmax(1).cpu().tolist()); probs.extend(z.softmax(1).max(1).values.cpu().tolist()); ims.extend(x.cpu())
 return loss_sum/n,correct/n,(ys,ps,probs,ims)
start=0; best=-1; history=[]; stale=0
if a.resume:
 ck=torch.load(a.resume,map_location=device,weights_only=False); assert all(ck['config'][k]==getattr(a,k) for k in ['model','seed','smoke','epochs']), '恢复配置不一致'; m.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler']); start=ck['epoch']+1; best=ck['best']; history=ck['history']; random.setstate(ck['random']); np.random.set_state(ck['numpy']); torch.set_rng_state(ck['torch'].cpu())
 if device=='cuda' and ck['cuda'] is not None: torch.cuda.set_rng_state_all(ck['cuda'])
if not a.test:
 for epoch in range(start,a.epochs):
  t=time.perf_counter(); tl,ta,_=step(tr,True); vl,vacc,_=step(va); lr=opt.param_groups[0]['lr']; sched.step(); improved=vacc>best; best=max(best,vacc); stale=0 if improved else stale+1
  history.append(dict(epoch=epoch,train_loss=tl,train_accuracy=ta,val_loss=vl,val_accuracy=vacc,lr=lr,seconds=time.perf_counter()-t))
  ck=dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),epoch=epoch,best=best,config=vars(a),history=history,random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if device=='cuda' else None)
  torch.save(ck,out/'last.pt')
  if improved: torch.save(ck,out/'best.pt')
  (out/'history.json').write_text(json.dumps(history,indent=2)); print(history[-1],flush=True)
  if a.patience and stale>=a.patience: break
 fig,axes=plt.subplots(1,2,figsize=(9,3))
 for ax,key in zip(axes,['loss','accuracy']):
  for split in ['train','val']: ax.plot([r[split+'_'+key] for r in history],label=split)
  ax.set_title(key); ax.legend()
 fig.tight_layout(); fig.savefig(out/'curves.png'); plt.close(fig)
else:
 ck=torch.load(a.resume or out/'best.pt',map_location=device,weights_only=False); m.load_state_dict(ck['model']); loss,acc,(ys,ps,probs,ims)=step(testset(a.data,base,a.smoke)); cm=confusion_matrix(ys,ps,labels=list(range(10))); norm=cm/np.maximum(cm.sum(1,keepdims=True),1); np.savetxt(out/'confusion_counts.csv',cm,fmt='%d',delimiter=','); np.savetxt(out/'confusion_normalized.csv',norm,delimiter=',')
 assert ck['config']['model']==a.model and ck['config']['smoke']==a.smoke
 metrics=dict(smoke=a.smoke,accuracy=acc,per_class_accuracy=norm.diagonal().tolist(),parameters=sum(x.numel() for x in m.parameters()),best_validation=ck['best'],training_seconds=sum(r['seconds'] for r in ck['history']),report=classification_report(ys,ps,labels=list(range(10)),output_dict=True,zero_division=0)); (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
 for arr,name in [(cm,'counts'),(norm,'normalized')]:
  fig,ax=plt.subplots(); im=ax.imshow(arr); fig.colorbar(im); ax.set(xlabel='Predicted',ylabel='True'); fig.savefig(out/('confusion_'+name+'.png')); plt.close(fig)
 ids=[i for i,(y,z) in enumerate(zip(ys,ps)) if y!=z][:9]; fig,axes=plt.subplots(3,3)
 for ax in axes.flat: ax.axis('off')
 for ax,i in zip(axes.flat,ids): ax.imshow((ims[i]*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None]).permute(1,2,0).clamp(0,1)); ax.set_title(f'T:{ys[i]} P:{ps[i]} {probs[i]:.2f}',fontsize=8)
 fig.tight_layout(); fig.savefig(out/'errors.png'); plt.close(fig); print(metrics)
(out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,config=vars(a)),indent=2))
