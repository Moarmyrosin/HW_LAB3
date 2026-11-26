"""先真实 smoke 验证，再三组训练，按验证集选模型，最后统一测试。"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from main import main as experiment
from utils import save_json
from data import md5_file, split_path


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--phase', choices=['train', 'evaluate', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data', default='data')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    experiment(['--audit', '--data', args.data, '--dataset', args.dataset])
    root = Path('runs') if args.dataset == 'cifar10' else Path('runs/cifar100')
    verification = Path(f'reports/verification_real_{args.dataset}.json')
    if args.phase in ['train', 'all']:
        if not verification.exists():
            raise RuntimeError('请先执行 python verify.py --real --dataset 对应数据集，确认真实数据恢复验证通过')
        checked = json.loads(verification.read_text(encoding='utf-8'))
        if checked.get('data_fingerprint') != md5_file(split_path(args.data, 42, args.dataset)):
            raise RuntimeError('真实 smoke 使用了不同的数据划分')
        if set(checked['results']) != {'cnn', 'resnet18', 'se'}:
            raise RuntimeError('真实 smoke 记录不完整')
        for name in ['cnn', 'resnet18', 'se']:
            out = root / name
            command = ['--dataset', args.dataset, '--model', name, '--epochs', str(args.epochs), '--batch_size', str(args.batch_size),
                       '--seed', '42', '--data', args.data, '--device', args.device, '--out', str(out)]
            if args.amp:
                command.append('--amp')
            if (out / 'last.pt').exists():
                command.extend(['--resume', str(out / 'last.pt')])
            experiment(command)
            progress = json.loads((out / 'progress.json').read_text(encoding='utf-8'))
            if progress['status'] != 'completed' or progress['epochs_completed'] != args.epochs:
                raise RuntimeError(f'{name} 尚未完成正式预算，停止后续统一测试')
    if args.phase in ['evaluate', 'all']:
        candidates = []
        for name in ['cnn', 'resnet18', 'se']:
            out = root / name
            ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
            if ck['config']['smoke'] or ck['epoch'] + 1 != ck['config']['epochs']:
                raise RuntimeError(f'{name} 正式训练尚未完成')
            candidates.append({'model': name, 'best_validation': ck['best'], 'best_epoch': ck['best_epoch'] + 1})
        # 固定并记录模型选择，再读取测试集评估，不根据测试成绩重新选择。
        selected = max(candidates, key=lambda row: row['best_validation'])['model']
        selection = {'criterion': 'maximum best validation accuracy; ties follow cnn/resnet18/se order',
                     'selected_model': selected, 'candidates': candidates}
        record = Path('reports/model_selection.json') if args.dataset == 'cifar10' else Path('reports/model_selection_cifar100.json')
        if record.exists() and json.loads(record.read_text(encoding='utf-8')) != selection:
            raise RuntimeError('已有模型选择记录不同，需核查实验而非覆盖')
        save_json(record, selection)
        for name in ['cnn', 'resnet18', 'se']:
            out = root / name
            if (out / 'metrics.json').exists():
                old = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
                last = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
                if old['run_id'] != last['run_id']:
                    raise RuntimeError('已有测试来自其他实验')
                print(f'{name} 已有同一实验的测试记录，保留结果', flush=True)
                continue
            experiment(['--dataset', args.dataset, '--model', name, '--test', '--data', args.data, '--out', str(out),
                        '--device', args.device, '--batch_size', str(args.batch_size)])
        print(f'按验证集选定模型：{selected}', flush=True)


if __name__ == '__main__':
    run()
