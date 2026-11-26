"""训练曲线、混淆矩阵与反归一化图像。英文类别名避免字体缺失。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from data import CLASSES, MEAN, STD


def denormalize(image, mean=MEAN, std=STD):
    return (image.cpu() * torch.tensor(std)[:, None, None] +
            torch.tensor(mean)[:, None, None]).permute(1, 2, 0).clamp(0, 1).numpy()


def curves(history, out):
    if not history:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ['loss', 'accuracy']):
        for split in ['train', 'val']:
            ax.plot([r['epoch'] + 1 for r in history], [r[split + '_' + key] for r in history], label=split)
        ax.set(xlabel='Epoch', ylabel=key.capitalize()); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(Path(out) / 'curves.png', dpi=160); plt.close(fig)


def confusion(counts, out, classes=CLASSES):
    count = len(classes)
    norm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    np.savetxt(Path(out) / 'confusion_counts.csv', counts, fmt='%d', delimiter=',')
    np.savetxt(Path(out) / 'confusion_normalized.csv', norm, delimiter=',')
    for values, suffix in [(counts, 'counts'), (norm, 'normalized')]:
        fig, ax = plt.subplots(figsize=(10, 8) if count <= 20 else (24, 22)); im = ax.imshow(values, cmap='Blues')
        fig.colorbar(im, ax=ax)
        ax.set(xticks=range(count), yticks=range(count), xticklabels=classes, yticklabels=classes,
               xlabel='Predicted class', ylabel='True class', title=f'Confusion matrix ({suffix})')
        plt.setp(ax.get_xticklabels(), rotation=90 if count > 20 else 45, ha='right', fontsize=5 if count > 20 else 9)
        plt.setp(ax.get_yticklabels(), fontsize=5 if count > 20 else 9)
        for i in range(count if count <= 20 else 0):
            for j in range(count):
                label = str(int(values[i, j])) if suffix == 'counts' else f'{values[i,j]:.2f}'
                ax.text(j, i, label, ha='center', va='center', fontsize=8,
                        color='white' if values[i, j] > values.max() / 2 else 'black')
        fig.tight_layout(); fig.savefig(Path(out) / f'confusion_{suffix}.png', dpi=160); plt.close(fig)
    return norm


def errors(images, targets, predictions, confidences, out, classes=CLASSES, mean=MEAN, std=STD):
    indices = [i for i, (true, pred) in enumerate(zip(targets, predictions)) if true != pred][:9]
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    for ax in axes.flat:
        ax.axis('off')
    for ax, index in zip(axes.flat, indices):
        ax.imshow(denormalize(images[index], mean, std), interpolation='nearest')
        ax.set_title(f'#{index}: {classes[targets[index]]}\nPred: {classes[predictions[index]]} ({confidences[index]:.1%})', fontsize=9)
    fig.suptitle('First misclassified examples in fixed test order' if indices else 'No errors in this evaluated subset')
    fig.tight_layout(); fig.savefig(Path(out) / 'errors.png', dpi=160); plt.close(fig)
    return [{'test_index': i, 'true': classes[targets[i]], 'predicted': classes[predictions[i]],
             'confidence': float(confidences[i])} for i in indices]
