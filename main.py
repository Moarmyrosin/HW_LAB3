"""命令行训练/恢复/独立测试入口；正式实验与 smoke 严格区分。"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from data import load, testset, audit, make_transforms, prepare, md5_file, split_path
from utils import save_json
from engine import train, evaluate


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', help='可选 JSON 配置；显式命令行参数优先')
    p.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    p.add_argument('--model', choices=['cnn', 'resnet18', 'se'], default='cnn')
    for name, default, kind in [('epochs', 30, int), ('batch_size', 128, int), ('seed', 42, int),
                                ('data', 'data', str), ('out', 'runs/cnn', str),
                                ('resume', '', str), ('checkpoint', '', str), ('device', 'auto', str),
                                ('patience', 0, int), ('num_workers', 0, int),
                                ('lr', .1, float), ('momentum', .9, float), ('weight_decay', .0005, float),
                                ('stop_after_epoch', 0, int)]:
        p.add_argument('--' + name, default=default, type=kind)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--test', action='store_true')
    p.add_argument('--audit', action='store_true', help='仅检查数据和划分')
    return p


def parse_args(argv=None):
    p = parser(); known, _ = p.parse_known_args(argv)
    if known.config:
        config = json.loads(Path(known.config).read_text(encoding='utf-8'))
        config.pop('optimizer', None)
        valid = {action.dest for action in p._actions}
        if set(config) - valid:
            raise ValueError(f'未知配置项：{set(config) - valid}')
        p.set_defaults(**config)
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.patience < 0:
        p.error('epochs、batch_size 必须为正，num_workers、patience 必须非负')
    args.data = str(Path(args.data).resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(4)
    if args.audit:
        info = audit(args.data, args.seed, args.dataset)
        save_json(Path('reports') / f'data_audit_{args.dataset}.json', info)
        print(json.dumps(info, ensure_ascii=False, indent=2)); return
    if args.test:
        prepare(args.data, args.dataset)
        split = split_path(args.data, args.seed, args.dataset)
        if not split.exists():
            raise FileNotFoundError('没有训练划分索引，不能核验 checkpoint 对应的数据')
        args.data_fingerprint = md5_file(split)
        _, base = make_transforms(args.dataset)
        return evaluate(args, testset(args.data, base, args.smoke, args.dataset))
    tr, va, _ = load(args.data, args.seed, args.smoke, args.dataset)
    args.data_fingerprint = md5_file(split_path(args.data, args.seed, args.dataset))
    args.classes = tr.dataset.classes
    return train(args, tr, va)


if __name__ == '__main__':
    main()
