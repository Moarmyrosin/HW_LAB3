"""从已有正式实验记录生成提交版紧凑图表，不重新训练。"""
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'reports'/'compact_figures'
COLORS={'cnn':'#637588','resnet18':'#217b92','se':'#c87531'}
NAMES={'cnn':'CNN','resnet18':'ResNet18','se':'SE-ResNet18'}


def load_metrics():
    return {n:json.loads((ROOT/'runs'/n/'metrics.json').read_text(encoding='utf-8')) for n in NAMES}


def raw_test():
    with (ROOT/'data/cifar-10-batches-py/test_batch').open('rb') as f:
        batch=pickle.load(f,encoding='latin1')
    return np.asarray(batch['data']).reshape(-1,3,32,32).transpose(0,2,3,1), np.asarray(batch['labels'])


def make_figures():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white','axes.labelcolor':'#253446',
                         'text.color':'#253446'})
    records=load_metrics()
    for name,r in records.items():
        assert not r['smoke'] and r['dataset']=='cifar10' and r['epochs_completed']==30
    fig,axes=plt.subplots(2,2,figsize=(7.05,3.9),layout='constrained')
    for ax,(key,title) in zip(axes.flat,[('train_loss','Training loss'),('val_loss','Validation loss'),
                                         ('train_accuracy','Training accuracy'),('val_accuracy','Validation accuracy')]):
        for name in NAMES:
            history=json.loads((ROOT/'runs'/name/'history.json').read_text(encoding='utf-8'))
            values=[row[key] for row in history]
            if 'accuracy' in key: values=np.array(values)*100
            ax.plot(range(1,31),values,label=NAMES[name],color=COLORS[name],lw=1.4)
        ax.set_title(title,fontsize=10.5,loc='left',fontweight='bold')
        ax.set_xlabel('Epoch',fontsize=9)
        if 'accuracy' in key: ax.set_ylabel('Accuracy (%)',fontsize=9)
        else: ax.set_ylabel('Cross-entropy',fontsize=9)
        ax.grid(alpha=.17); ax.tick_params(labelsize=8)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.savefig(FIG/'learning_curves.png',dpi=300); plt.close(fig)

    selected=json.loads((ROOT/'reports/model_selection.json').read_text(encoding='utf-8'))['selected_model']
    r=records[selected]; classes=r['classes']
    counts=np.loadtxt(ROOT/'runs'/selected/'confusion_counts.csv',delimiter=',')
    normalized=np.loadtxt(ROOT/'runs'/selected/'confusion_normalized.csv',delimiter=',')
    fig,axes=plt.subplots(1,2,figsize=(10,4.65),layout='constrained')
    labels=['plane','auto','bird','cat','deer','dog','frog','horse','ship','truck']
    for ax,values,title in zip(axes,[counts,normalized],['(a) Raw counts','(b) Row-normalized (%)']):
        ax.imshow(values,cmap='Blues',vmin=0,vmax=1000 if values is counts else 1)
        ax.set(xticks=range(10),yticks=range(10),xticklabels=labels,yticklabels=labels)
        ax.set_title(title,fontsize=13,loc='left')
        ax.set_xlabel('Predicted class',fontsize=12); ax.set_ylabel('True class',fontsize=12)
        ax.tick_params(axis='both',labelsize=11)
        plt.setp(ax.get_xticklabels(),rotation=55,ha='right')
        for i in range(10):
            for j in range(10):
                text=str(int(values[i,j])) if values is counts else f'{values[i,j]*100:.0f}'
                ax.text(j,i,text,ha='center',va='center',fontsize=11.5,
                        color='white' if values[i,j]>.55*values.max() else '#253446')
    fig.savefig(FIG/'confusion_selected.png',dpi=300); plt.close(fig)

    images,truth=raw_test()
    pred=np.load(ROOT/'runs'/selected/'predictions.npz')
    ids=[15,25,35,37,58,78]
    fig,axes=plt.subplots(2,3,figsize=(7.05,3.9),layout='constrained')
    for ax,index in zip(axes.flat,ids):
        predicted=int(pred['predictions'][index]); conf=float(pred['confidences'][index])
        assert predicted!=int(truth[index])
        ax.imshow(images[index],interpolation='nearest'); ax.axis('off')
        ax.set_title(f'#{index} {classes[int(truth[index])]} → {classes[predicted]}\np={conf:.1%}',fontsize=9,pad=4)
    fig.savefig(FIG/'errors_selected.png',dpi=300); plt.close(fig)

    # 重算同一最佳权重的固定解释案例，改变排版而不改变选择规则。
    import torch
    from models import build
    from gradcam import target_layer
    from data import MEAN,STD
    torch.set_num_threads(2)
    ck=torch.load(ROOT/'runs'/selected/'best.pt',map_location='cpu',weights_only=False)
    assert ck['run_id']==r['run_id']
    model=build(selected).eval(); model.load_state_dict(ck['model'])
    original=json.loads((ROOT/'runs'/selected/'gradcam.json').read_text(encoding='utf-8'))
    cases=original['examples']
    cache={}
    def hook(module,inputs,output):
        cache['features']=output; output.retain_grad()
    handle=target_layer(model).register_forward_hook(hook)
    fig=plt.figure(figsize=(7.05,3.75))
    outer=GridSpec(2,3,figure=fig,left=.01,right=.99,top=.91,bottom=.02,wspace=.12,hspace=.40)
    metadata=[]
    for cell,example in zip(outer,cases):
        index=example['official_test_index']; x=torch.tensor(images[index].transpose(2,0,1).copy(),dtype=torch.float32)/255
        x=(x-torch.tensor(MEAN)[:,None,None])/torch.tensor(STD)[:,None,None]
        restored=(x*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None]).permute(1,2,0).clamp(0,1).numpy()
        assert np.max(np.abs(restored-images[index]/255.0))<1e-6
        model.zero_grad(set_to_none=True); logits=model(x.unsqueeze(0))
        target=logits.argmax(1).item()
        assert classes[target]==example['predicted']
        confidence=logits.softmax(1)[0,target].item()
        logits[0,target].backward(); feature=cache['features']
        cam=(feature.grad.mean((2,3),keepdim=True)*feature).sum(1,keepdim=True).relu()
        cam=torch.nn.functional.interpolate(cam,(32,32),mode='bilinear',align_corners=False)[0,0]
        cam=(cam/(cam.max()+1e-8)).detach().numpy()
        inner=GridSpecFromSubplotSpec(1,2,subplot_spec=cell,wspace=.03)
        a=fig.add_subplot(inner[0]); b=fig.add_subplot(inner[1])
        for ax in [a,b]: ax.imshow(restored,interpolation='nearest'); ax.axis('off')
        b.imshow(cam,cmap='jet',alpha=.45,vmin=0,vmax=1)
        a.set_title('Input',fontsize=7.8,pad=2); b.set_title('Grad-CAM',fontsize=7.8,pad=2)
        bounds=cell.get_position(fig)
        label=f'#{index} {example["true"]} → {example["predicted"]} ({confidence:.1%})'
        fig.text(bounds.x0,bounds.y1+.055,label,fontsize=8.3,ha='left')
        metadata.append({'index':index,'true':example['true'],'predicted':example['predicted'],
                         'confidence':confidence,'feature_shape':list(feature.shape)})
    handle.remove()
    fig.savefig(FIG/'gradcam_selected.png',dpi=300); plt.close(fig)
    (FIG/'figure_provenance.json').write_text(json.dumps({
        'source':'existing CIFAR-10 formal runs; no retraining',
        'formal_run_ids':{n:r['run_id'] for n,r in records.items()},
        'selected_model':selected,'error_indices':ids,'gradcam_cases':metadata,
        'error_confidence_precision':'original AMP test','gradcam_precision':'float32 CPU'},
        ensure_ascii=False,indent=2),encoding='utf-8')
    print('已生成 4 张紧凑科学图表：',FIG)


if __name__=='__main__':
    make_figures()
