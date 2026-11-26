"""训练、恢复与独立评估核心；最佳模型由验证集选择。"""
import csv
import json
import time
import uuid
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from models import build
from data import CLASSES, dataset_spec
from utils import (save_json, seed_all, rng_state, restore_rng,
                   environment, choose_device, checkpoint_save)
from plots import curves, confusion, errors

RESUME_KEYS = ['dataset', 'classes', 'model', 'epochs', 'batch_size', 'seed', 'smoke', 'amp',
               'lr', 'momentum', 'weight_decay', 'patience', 'num_workers',
               'data', 'data_fingerprint']


def loader(dataset, args, device, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                      generator=generator)


def epoch_step(model, batches, device, optimizer=None, scaler=None, amp=False, collect=False):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.int64)
    total = 0
    targets, predictions, confidences, images = [], [], [], []
    with torch.set_grad_enabled(training):
        for x, y in batches:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('发现非有限 loss，停止训练，保留上一轮 checkpoint')
            if training:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += loss.detach() * len(y)
            correct += (logits.argmax(1) == y).sum()
            total += len(y)
            if collect:
                prob = logits.detach().float().softmax(1)
                targets.extend(y.cpu().tolist())
                predictions.extend(prob.argmax(1).cpu().tolist())
                confidences.extend(prob.max(1).values.cpu().tolist())
                images.extend(x.detach().cpu())
    if total == 0:
        raise ValueError('数据子集为空')
    return float(loss_sum.item() / total), float(correct.item() / total), (targets, predictions, confidences, images)


def save_history(history, out):
    save_json(out / 'history.json', history)
    if history:
        with (out / 'history.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)


def train(args, trainset, valset):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((out / name).exists() for name in ['last.pt', 'best.pt', 'train_environment.json']):
        raise FileExistsError(f'已有实验记录，拒绝覆盖：{out}；请恢复训练或使用新目录')
    device = choose_device(args.device)
    seed_all(args.seed)
    model = build(args.model, len(args.classes)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = loader(trainset, args, device, True, generator)
    val_loader = loader(valset, args, device)
    history, start, best, stale, best_epoch = [], 0, -1., 0, -1
    run_id = str(uuid.uuid4())
    config = {k: getattr(args, k) for k in RESUME_KEYS}
    if args.resume:
        source = Path(args.resume)
        if source.parent.resolve() != out.resolve():
            raise ValueError('恢复训练须写回原实验目录，以保留最佳权重和完整历史')
        ck = torch.load(source, map_location='cpu', weights_only=False)
        if ck.get('schema') != 2:
            raise ValueError('旧 checkpoint 版本不同，保留旧文件并使用新目录重新运行')
        for key in RESUME_KEYS:
            if ck['config'].get(key) != config.get(key):
                raise ValueError(f'恢复配置不一致：{key}，原值 {ck["config"].get(key)}，当前 {config.get(key)}')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        history = ck['history']; start = ck['epoch'] + 1; best = ck['best']
        stale = ck['stale']; best_epoch = ck['best_epoch']; run_id = ck['run_id']
        generator.set_state(ck['loader_generator'].cpu()); restore_rng(ck['rng'])
        save_json(out / f'resume_environment_epoch_{start}.json', environment(device))
    else:
        save_json(out / 'config.json', config)
        save_json(out / 'train_environment.json', environment(device))
    status = 'running'
    if args.patience and stale >= args.patience:
        status = 'early_stopped'
    try:
        for epoch in range(start, args.epochs):
            if status == 'early_stopped':
                break
            if device.type == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter()
            tl, ta, _ = epoch_step(model, train_loader, device, optimizer, scaler, args.amp)
            vl, va, _ = epoch_step(model, val_loader, device, amp=args.amp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            lr = optimizer.param_groups[0]['lr']; scheduler.step()
            improved = va > best
            if improved:
                best, best_epoch, stale = va, epoch, 0
            else:
                stale += 1
            history.append(dict(epoch=epoch, train_loss=tl, train_accuracy=ta,
                                val_loss=vl, val_accuracy=va, lr=lr, seconds=elapsed))
            state = dict(schema=2, run_id=run_id, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict(), epoch=epoch, best=best,
                         best_epoch=best_epoch, stale=stale, config=config,
                         history=history, rng=rng_state(), loader_generator=generator.get_state())
            checkpoint_save(out / 'last.pt', state)
            if improved:
                checkpoint_save(out / 'best.pt', state)
            save_history(history, out)
            save_json(out / 'progress.json', dict(status='running', epochs_completed=epoch + 1,
                      epochs_planned=args.epochs, best_validation=best, best_epoch=best_epoch + 1,
                      training_seconds=sum(row['seconds'] for row in history), run_id=run_id))
            print(f'{args.model} epoch {epoch+1}/{args.epochs}: train={ta:.4f}, val={va:.4f}, loss={vl:.4f}, seconds={elapsed:.1f}', flush=True)
            if args.patience and stale >= args.patience:
                status = 'early_stopped'; break
            # 仅供验证中断恢复：不更改总预算或余弦调度周期。
            if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch:
                status = 'paused_for_verification'; break
        else:
            status = 'completed'
    except KeyboardInterrupt:
        status = 'interrupted'
        print('训练被中断；上一完整轮的 last.pt 可用于恢复。', flush=True)
    finally:
        save_history(history, out); curves(history, out)
        save_json(out / 'progress.json', dict(status=status, epochs_completed=len(history),
                  epochs_planned=args.epochs, best_validation=best, best_epoch=best_epoch + 1,
                  training_seconds=sum(row['seconds'] for row in history), run_id=run_id,
                  time_definition='所有完整 epoch 的训练与验证耗时之和，不包含数据准备、绘图和 checkpoint 写盘'))
    return history


def validate_checkpoint(ck, args):
    if ck.get('schema') != 2:
        raise ValueError('checkpoint 版本不兼容')
    for key in ['dataset', 'model', 'seed', 'smoke']:
        if ck['config'][key] != getattr(args, key):
            raise ValueError(f'评估配置不一致：{key}')
    if ck['epoch'] != ck['best_epoch']:
        raise ValueError('独立测试必须使用最佳验证 checkpoint，请指定 best.pt')


def evaluate(args, dataset):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = Path(args.checkpoint or args.resume or out / 'best.pt')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint(ck, args)
    last_path = path.parent / 'last.pt'
    if not last_path.exists():
        raise FileNotFoundError('缺少 last.pt，无法核验完整训练耗时')
    last = torch.load(last_path, map_location='cpu', weights_only=False)
    if last['run_id'] != ck['run_id'] or last['config'] != ck['config']:
        raise ValueError('最佳 checkpoint 与最后 checkpoint 不属于同一实验')
    if last['best_epoch'] != ck['epoch']:
        raise ValueError('指定 checkpoint 不是当前实验的最佳权重')
    if args.data_fingerprint != ck['config']['data_fingerprint']:
        raise ValueError('评估数据划分指纹不一致')
    if (out / 'metrics.json').exists():
        raise FileExistsError(f'已有测试结果，拒绝重复覆盖：{out}')
    device = choose_device(args.device)
    seed_all(args.seed)
    classes = ck['config']['classes']
    model = build(args.model, len(classes)).to(device); model.load_state_dict(ck['model'])
    # 测试用训练时配置的 AMP 设置，避免无意改变评估数值路径。
    amp = ck['config']['amp'] and device.type == 'cuda'
    loss, acc, (targets, preds, conf, images) = epoch_step(model, loader(dataset, args, device), device, amp=amp, collect=True)
    counts = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    assert counts.sum() == len(dataset)
    norm = confusion(counts, out, classes)
    spec = dataset_spec(args.dataset)
    examples = errors(images, targets, preds, conf, out, classes, spec['mean'], spec['std'])
    for example in examples:
        index = example['test_index']
        example['official_test_index'] = int(dataset.indices[index]) if hasattr(dataset, 'indices') else index
    report = classification_report(targets, preds, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    for i, name in enumerate(classes):
        assert abs(report[name]['recall'] - norm[i, i]) < 1e-12
    np.savez_compressed(out / 'predictions.npz', targets=targets, predictions=preds, confidences=conf)
    save_json(out / 'classification_report.json', report)
    (out / 'classification_report.txt').write_text(
        classification_report(targets, preds, labels=list(range(len(classes))), target_names=classes, zero_division=0), encoding='utf-8')
    metrics = dict(smoke=args.smoke, source=f'official {args.dataset}', dataset=args.dataset, model=args.model,
                   run_id=ck['run_id'], checkpoint=str(path.resolve()), checkpoint_epoch=ck['epoch'] + 1,
                   accuracy=acc, loss=loss, samples=len(dataset), classes=classes,
                   per_class_accuracy=norm.diagonal().tolist(), report=report,
                   parameters=sum(p.numel() for p in model.parameters()), best_validation=ck['best'],
                   epochs_completed=len(last['history']), epochs_planned=ck['config']['epochs'],
                   training_seconds=sum(row['seconds'] for row in last['history']),
                   config=ck['config'], error_examples=examples)
    save_json(out / 'metrics.json', metrics)
    save_json(out / 'test_environment.json', environment(device))
    save_json(out / 'test_config.json', {'checkpoint': str(path.resolve()), 'training_config': ck['config'],
                                       'eval_batch_size': args.batch_size, 'amp_effective': amp})
    print(f'{args.model} test accuracy={acc:.4f}, n={len(dataset)}; smoke={args.smoke}', flush=True)
    return metrics
