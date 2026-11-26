"""先用标准库校验数据包；成功后才导入 torchvision 解压和审计划分。"""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = 'c58f30108f718f92721af3b95e74349a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    args = parser.parse_args()
    expected = EXPECTED if args.dataset == 'cifar10' else 'eb9058c3a382ffc7106e4002c42a8d85'
    archive = Path(args.data).resolve() / ('cifar-10-python.tar.gz' if args.dataset == 'cifar10' else 'cifar-100-python.tar.gz')
    if args.dataset == 'cifar100' and not archive.exists():
        alternate = Path(args.data).resolve() / 'cifar-10-python.tar.gz'
        if alternate.exists(): archive = alternate
    result = {'archive': str(archive), 'exists': archive.is_file(), 'expected_md5': expected, 'dataset': args.dataset}
    if archive.is_file():
        digest = hashlib.md5()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        result['actual_md5'] = digest.hexdigest()
        result['valid'] = result['actual_md5'] == expected
        result['reason'] = ('校验通过' if result['valid'] else
                            '文件为 CIFAR-100 Python 包，不能用于本作业' if result['actual_md5'] == 'eb9058c3a382ffc7106e4002c42a8d85' else '文件不完整或不是 CIFAR-10 Python 包')
    else:
        result.update(valid=False, reason='缺少压缩包，请放置到上述 archive 路径')
    Path('reports').mkdir(exist_ok=True)
    Path(f'reports/data_package_check_{args.dataset}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result['valid']:
        raise SystemExit(2)
    from data import audit
    info = audit(args.data, dataset=args.dataset)
    Path(f'reports/data_audit_{args.dataset}.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
