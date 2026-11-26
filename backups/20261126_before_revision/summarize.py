"""汇总真实运行输出；缺失项保留待运行。"""
import json
from pathlib import Path
rows=['|模型|参数量|最佳验证准确率|测试准确率|训练秒数|','|---|---|---|---|---|']
for name in ['cnn','resnet18','se']:
 p=Path('runs')/name/'metrics.json'
 if p.exists():
  r=json.loads(p.read_text()); assert not r['smoke']; rows.append(f"|{name}|{r['parameters']}|{r['best_validation']:.4f}|{r['accuracy']:.4f}|{r['training_seconds']:.1f}|")
 else: rows.append(f'|{name}|待运行|待运行|待运行|待运行|')
Path('reports').mkdir(exist_ok=True); Path('reports/comparison.md').write_text('\n'.join(rows),encoding='utf-8')
