import json,tempfile
from pathlib import Path
import torch
from models import build
torch.set_num_threads(2); torch.manual_seed(42); results={}
for name in ['cnn','resnet18','se']:
 m=build(name); opt=torch.optim.SGD(m.parameters(),lr=.01); x=torch.randn(4,3,32,32); y=torch.tensor([0,1,2,3]); z=m(x); assert z.shape==(4,10); loss=torch.nn.functional.cross_entropy(z,y); loss.backward(); opt.step(); m.eval()
 with torch.no_grad(): before=m(x)
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'check.pt'; torch.save(m.state_dict(),path); other=build(name); other.load_state_dict(torch.load(path,weights_only=True)); other.eval()
  with torch.no_grad(): assert torch.equal(before,other(x))
 results[name]={'shape':list(z.shape),'synthetic_loss':loss.item(),'save_load_equal':True,'parameters':sum(p.numel() for p in m.parameters())}
Path('verification.json').write_text(json.dumps(results,indent=2)); print(results)
