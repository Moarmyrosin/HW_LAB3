"""只从正式实验文件汇总结果，并核验历史、预测、混淆矩阵一致。"""
import json
from pathlib import Path
import numpy as np
from utils import save_json


def collect(dataset='cifar10'):
    root = Path('runs') if dataset == 'cifar10' else Path('runs/cifar100')
    result = {}
    for name in ['cnn', 'resnet18', 'se']:
        out = root / name
        if not (out / 'metrics.json').exists():
            continue
        r = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
        history = json.loads((out / 'history.json').read_text(encoding='utf-8'))
        if r['smoke'] or r['model'] != name or r['dataset'] != dataset:
            raise ValueError('正式目录中出现 smoke 或模型不一致')
        assert r['epochs_completed'] == len(history)
        assert r['best_validation'] == max(row['val_accuracy'] for row in history)
        assert abs(r['training_seconds'] - sum(row['seconds'] for row in history)) < 1e-6
        counts = np.loadtxt(out / 'confusion_counts.csv', delimiter=',')
        norm = np.loadtxt(out / 'confusion_normalized.csv', delimiter=',')
        assert counts.sum() == r['samples'] == 10000
        assert np.isclose(counts.trace() / counts.sum(), r['accuracy'])
        assert np.allclose(norm, counts / counts.sum(1, keepdims=True))
        assert np.allclose(norm.diagonal(), r['per_class_accuracy'])
        with np.load(out / 'predictions.npz') as predictions:
            assert np.isclose(np.mean(predictions['targets'] == predictions['predictions']), r['accuracy'])
        r['top_confusions'] = [
            {'true': r['classes'][i], 'predicted': r['classes'][j], 'count': int(counts[i, j]),
             'fraction_of_true_class': float(norm[i, j])}
            for i, j in sorted(((i, j) for i in range(len(r['classes'])) for j in range(len(r['classes'])) if i != j),
                               key=lambda pair: counts[pair], reverse=True)[:5]]
        result[name] = r
    return result


def summary_table(results):
    rows = ['|模型|参数量|最佳验证准确率|测试准确率|完成轮数|训练与验证耗时（秒）|',
            '|---|---:|---:|---:|---:|---:|']
    for name in ['cnn', 'resnet18', 'se']:
        if name in results:
            r = results[name]
            rows.append(f'|{name}|{r["parameters"]:,}|{r["best_validation"]:.2%}|{r["accuracy"]:.2%}|{r["epochs_completed"]}/{r["epochs_planned"]}|{r["training_seconds"]:.1f}|')
        else:
            rows.append(f'|{name}|待运行|待运行|待运行|待运行|待运行|')
    return '\n'.join(rows)


def main():
    results = collect()
    Path('reports').mkdir(exist_ok=True)
    (Path('reports') / 'comparison.md').write_text(summary_table(results) + '\n', encoding='utf-8')
    save_json('reports/comparison.json', results)
    print(summary_table(results))


if __name__ == '__main__':
    main()
