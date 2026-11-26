"""对按验证集选定的模型生成正确/错误案例：原图与 Grad-CAM 并排。"""
import argparse
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import build, Block
from data import prepare, make_transforms, testset, CLASSES, dataset_spec
from plots import denormalize
from utils import choose_device, save_json


def target_layer(model):
    blocks = [module for module in model.modules() if isinstance(module, Block)]
    # SE 的门控卷积是 1×1；最后残差块输出仍为 4×4，适合空间解释。
    return blocks[-1] if blocks else [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)][-1]


def generate(model, dataset, device, output, max_each=3, classes=CLASSES, mean=None, std=None):
    model.eval()
    layer = target_layer(model)
    cache, examples, counts = {}, [], {'correct': 0, 'wrong': 0}
    def capture(module, inputs, features):
        cache['features'] = features
        features.retain_grad()
    handle = layer.register_forward_hook(capture)
    try:
        for index, (image, truth) in enumerate(dataset):
            model.zero_grad(set_to_none=True)
            logits = model(image.unsqueeze(0).to(device))
            probabilities = logits.float().softmax(1)
            prediction = logits.argmax(1).item()
            kind = 'correct' if prediction == int(truth) else 'wrong'
            if counts[kind] >= max_each:
                continue
            logits[0, prediction].backward()
            features = cache['features']
            if min(features.shape[-2:]) <= 1:
                raise ValueError('Grad-CAM 目标层没有空间信息')
            weights = features.grad.mean((2, 3), keepdim=True)
            raw = (weights * features).sum(1, keepdim=True).relu()
            heatmap = torch.nn.functional.interpolate(raw, (32, 32), mode='bilinear', align_corners=False)[0, 0]
            heatmap = (heatmap / (heatmap.max() + 1e-8)).detach().cpu().numpy()
            examples.append({'test_index': index, 'official_test_index': int(dataset.indices[index]) if hasattr(dataset, 'indices') else index, 'true': classes[int(truth)],
                             'predicted': classes[prediction], 'confidence': probabilities[0, prediction].item(),
                             'kind': kind, 'feature_shape': list(features.shape), 'nonzero_cam': bool(raw.max().item() > 0),
                             'image': denormalize(image, mean, std) if mean is not None else denormalize(image), 'heatmap': heatmap})
            counts[kind] += 1
            if min(counts.values()) >= max_each:
                break
    finally:
        handle.remove()
    if not examples:
        raise ValueError('没有可视化案例')
    fig, axes = plt.subplots(len(examples), 2, figsize=(7, 3 * len(examples)), squeeze=False)
    for row, example in zip(axes, examples):
        for ax in row:
            ax.imshow(example['image'], interpolation='nearest'); ax.axis('off')
        row[0].set_title(f'#{example["test_index"]} True: {example["true"]}')
        row[1].imshow(example['heatmap'], cmap='jet', alpha=.45, vmin=0, vmax=1)
        row[1].set_title(f'{example["kind"]}: {example["predicted"]} ({example["confidence"]:.1%})')
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)
    record = {'selection': 'first three correct and first three wrong in fixed dataset order',
              'precision': 'float32', 'counts': counts,
              'examples': [{key: value for key, value in example.items() if key not in ['image', 'heatmap']}
                           for example in examples]}
    save_json(output.with_suffix('.json'), record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='省略时读取 reports/model_selection.json')
    parser.add_argument('--data', default='data')
    parser.add_argument('--out')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.checkpoint:
        selected = json.loads(Path('reports/model_selection.json').read_text(encoding='utf-8'))['selected_model']
        args.checkpoint = str(Path('runs') / selected / 'best.pt')
    checkpoint = Path(args.checkpoint)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck.get('schema') != 2 or ck['config']['smoke'] != args.smoke:
        raise ValueError('checkpoint 版本或 smoke 设置不符')
    if ck['epoch'] != ck['best_epoch']:
        raise ValueError('解释时请使用最佳验证模型 best.pt')
    dataset_name = ck['config']['dataset']
    spec = dataset_spec(dataset_name)
    prepare(args.data, dataset_name)
    _, base = make_transforms(dataset_name)
    dataset = testset(args.data, base, args.smoke, dataset_name)
    device = choose_device(args.device)
    model = build(ck['config']['model'], len(ck['config']['classes'])).to(device)
    model.load_state_dict(ck['model'])
    record = generate(model, dataset, device, args.out or checkpoint.parent / 'gradcam.png', classes=ck['config']['classes'], mean=spec['mean'], std=spec['std'])
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
