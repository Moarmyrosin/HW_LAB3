"""离线合成数据或 --real 官方数据的训练恢复检查；结果不计入正式性能。"""
import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import TensorDataset
from main import parse_args
from data import load, audit, md5_file, testset, split_path, CLASSES
from engine import train, evaluate, validate_checkpoint
from models import build, Block
from utils import save_json, choose_device

METRIC_KEYS = ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'lr']


def verify_model(name, root, tr, va, fingerprint, device, real, data_root="data", dataset="cifar10"):
    args = parse_args(['--model', name, '--epochs', '2', '--batch_size', '40',
                       '--smoke', '--out', str(root / name / 'continuous'), '--device', device])
    args.data = str(Path(data_root).resolve())
    args.dataset = dataset
    args.classes = tr.dataset.classes if real else CLASSES
    args.amp = device.startswith('cuda')
    args.data_fingerprint = fingerprint
    train(args, tr, va)
    reference = torch.load(Path(args.out) / 'last.pt', map_location='cpu', weights_only=False)
    interrupted = copy.deepcopy(args)
    interrupted.out = str(root / name / 'resumed')
    interrupted.stop_after_epoch = 1
    train(interrupted, tr, va)
    interrupted.resume = str(Path(interrupted.out) / 'last.pt')
    interrupted.stop_after_epoch = 0
    train(interrupted, tr, va)
    resumed = torch.load(interrupted.resume, map_location='cpu', weights_only=False)
    equal = all(torch.equal(reference['model'][key], resumed['model'][key]) for key in reference['model'])
    assert equal, f'{name} 断点恢复与连续训练权重不一致'
    for key in METRIC_KEYS:
        assert [r[key] for r in reference['history']] == [r[key] for r in resumed['history']], key
    assert reference['scheduler'] == resumed['scheduler']
    assert reference['scaler'] == resumed['scaler']
    assert reference['stale'] == resumed['stale'] and reference['best_epoch'] == resumed['best_epoch']
    ck = torch.load(Path(args.out) / 'best.pt', map_location='cpu', weights_only=False)
    invalid = copy.deepcopy(args); invalid.smoke = False
    try:
        validate_checkpoint(ck, invalid)
    except ValueError:
        rejected = True
    else:
        raise AssertionError('未拒绝 smoke/正式权重混用')
    model = build(name, len(args.classes)).to(device).eval()
    model.load_state_dict(ck['model'])
    blocks = [module for module in model.modules() if isinstance(module, Block)]
    target = blocks[-1] if blocks else [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)][-1]
    cache = {}
    def hook(module, inputs, output):
        cache['f'] = output; output.retain_grad()
    handle = target.register_forward_hook(hook)
    x, _ = va[0]; logits = model(x.unsqueeze(0).to(device))
    assert logits.shape == (1, len(args.classes))
    logits[0, logits.argmax(1).item()].backward()
    features = cache['f']
    assert min(features.shape[-2:]) > 1 and features.grad is not None
    feature_shape = list(features.shape)
    handle.remove()
    result = {'output_shape': list(logits.shape), 'resume_weights_identical': equal,
              'resume_metrics_identical': True, 'scheduler_scaler_early_stop_state_equal': True,
              'reject_smoke_formal_mismatch': rejected, 'gradcam_feature_shape': feature_shape,
              'parameters': sum(p.numel() for p in model.parameters())}
    del model, reference, resumed, ck
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if real:
        _, _, base = load(args.data, args.seed, True, args.dataset)
        args.test = True
        env_bytes = (Path(args.out) / 'train_environment.json').read_bytes()
        metrics = evaluate(args, testset(args.data, base, True, args.dataset))
        assert (Path(args.out) / 'train_environment.json').read_bytes() == env_bytes
        assert metrics['epochs_completed'] == 2
        assert metrics['training_seconds'] == sum(r['seconds'] for r in json.loads((Path(args.out) / 'history.json').read_text(encoding='utf-8')))
        counts = np.loadtxt(Path(args.out) / 'confusion_counts.csv', delimiter=',')
        assert int(counts.sum()) == metrics['samples']
        result['real_smoke_metrics'] = {'accuracy': metrics['accuracy'], 'samples': metrics['samples']}
        result['train_environment_preserved_after_test'] = True
        result['full_training_time_verified'] = True
    return result


def verify_early_stop():
    """固定输入配合均衡标签，保证验证准确率恒定，检查非零 patience 状态恢复。"""
    torch.set_num_threads(2)
    tr = TensorDataset(torch.zeros(20, 3, 32, 32), torch.arange(20) % 10)
    va = TensorDataset(torch.zeros(10, 3, 32, 32), torch.arange(10))
    Path('tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir='tmp', prefix='early_stop_') as temporary:
        root = Path(temporary)
        args = parse_args(['--model', 'cnn', '--epochs', '4', '--lr', '0', '--patience', '2',
                           '--batch_size', '40', '--smoke', '--device', 'cpu', '--out', str(root / 'continuous')])
        args.classes = CLASSES; args.data_fingerprint = 'synthetic'
        train(args, tr, va)
        expected = torch.load(Path(args.out) / 'last.pt', map_location='cpu', weights_only=False)
        paused = copy.deepcopy(args); paused.out = str(root / 'resumed'); paused.stop_after_epoch = 2
        train(paused, tr, va)
        checkpoint = Path(paused.out) / 'last.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['stale'] == 1
        paused.resume = str(checkpoint); paused.stop_after_epoch = 0
        train(paused, tr, va)
        actual = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert actual['epoch'] == expected['epoch'] == 2
        assert actual['stale'] == expected['stale'] == 2
        assert all(torch.equal(expected['model'][key], actual['model'][key]) for key in expected['model'])
        progress = json.loads((Path(paused.out) / 'progress.json').read_text(encoding='utf-8'))
        assert progress['status'] == 'early_stopped'
    result = {'source': 'synthetic only', 'stale_at_interruption': 1,
              'patience': 2, 'epochs_completed': 3, 'epochs_planned': 4,
              'continuous_and_resumed_stop_at_same_epoch': True, 'weights_identical': True}
    save_json('reports/verification_early_stop.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--early-stop-only', action='store_true')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--data', default='data')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', default=None)
    options = parser.parse_args()
    if options.early_stop_only:
        return verify_early_stop()
    torch.set_num_threads(4)
    device = str(choose_device(options.device))
    if options.real:
        info = audit(options.data, dataset=options.dataset)
        save_json(f'reports/data_audit_{options.dataset}.json', info)
        tr, va, _ = load(options.data, 42, True, options.dataset)
        fingerprint = md5_file(split_path(options.data, 42, options.dataset))
        root = Path(options.out or f'runs/smoke_real_{options.dataset}'); root.mkdir(parents=True, exist_ok=True)
        results = {name: verify_model(name, root, tr, va, fingerprint, device, True, options.data, options.dataset)
                   for name in ['cnn', 'resnet18', 'se']}
        save_json(f'reports/verification_real_{options.dataset}.json', {'source': f'official {options.dataset} smoke', 'dataset': options.dataset, 'device': device, 'data_fingerprint': fingerprint, 'results': results})
    else:
        gen = torch.Generator().manual_seed(7)
        tr = TensorDataset(torch.randn(20, 3, 32, 32, generator=gen), torch.arange(20) % 10)
        va = TensorDataset(torch.randn(10, 3, 32, 32, generator=gen), torch.arange(10))
        Path('tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir='tmp', prefix='verify_') as temporary:
            results = {name: verify_model(name, Path(temporary), tr, va, 'synthetic', device, False)
                       for name in ['cnn', 'resnet18', 'se']}
        save_json('reports/verification_offline.json', {'source': 'synthetic only', 'device': device, 'results': results})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
