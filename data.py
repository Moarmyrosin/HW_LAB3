"""官方 CIFAR 数据校验、分层划分及独立变换；默认作业数据为 CIFAR-10。"""
import hashlib
from pathlib import Path
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import extract_archive

MEAN = (.4914, .4822, .4465)
STD = (.247, .2435, .2616)
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
ARCHIVE_MD5 = 'c58f30108f718f92721af3b95e74349a'
SPECS = {
    'cifar10': {'constructor': datasets.CIFAR10, 'num_classes': 10,
                'archive_name': 'cifar-10-python.tar.gz', 'md5': ARCHIVE_MD5,
                'mean': MEAN, 'std': STD},
    'cifar100': {'constructor': datasets.CIFAR100, 'num_classes': 100,
                 'archive_name': 'cifar-100-python.tar.gz', 'md5': 'eb9058c3a382ffc7106e4002c42a8d85',
                 'mean': (.5071, .4867, .4408), 'std': (.2675, .2565, .2761)}
}


def dataset_spec(name='cifar10'):
    if name not in SPECS:
        raise ValueError(f'不支持的数据集：{name}')
    return SPECS[name]


def split_path(root, seed, dataset='cifar10'):
    return Path(root) / (f'split_{seed}.npz' if dataset == 'cifar10' else f'split_{dataset}_{seed}.npz')


def archive_path(root, dataset='cifar10'):
    preferred = Path(root) / dataset_spec(dataset)['archive_name']
    # 用户提供的 CIFAR-100 包误命名为 CIFAR-10；仅在显式选择 cifar100 时兼容。
    if dataset == 'cifar100' and not preferred.exists():
        alternate = Path(root) / 'cifar-10-python.tar.gz'
        if alternate.exists():
            return alternate
    return preferred


def md5_file(path):
    digest = hashlib.md5()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root, dataset='cifar10'):
    """按实际数据集校验 MD5，校验通过后才解压，绝不自动下载。"""
    spec = dataset_spec(dataset)
    constructor = spec['constructor']
    root = Path(root)
    archive = archive_path(root, dataset)
    if archive.exists():
        actual = md5_file(archive)
        if actual != spec['md5']:
            hint = '（该校验值属于 CIFAR-100 Python 包）' if actual == SPECS['cifar100']['md5'] else ''
            raise ValueError(f'{dataset} 数据包不符{hint}：{archive.resolve()}\n实际 MD5={actual}\n要求 MD5={spec["md5"]}；已停止，未解压、未下载。')
    try:
        constructor(root, train=True, download=False)
        constructor(root, train=False, download=False)
    except RuntimeError:
        if not archive.exists():
            raise FileNotFoundError(f'请将完整官方 {dataset} Python 包放在 {archive.resolve()}')
        extract_archive(str(archive), str(root))
        constructor(root, train=True, download=False)
        constructor(root, train=False, download=False)
    return {'dataset': dataset, 'archive': str(archive.resolve()),
            'archive_md5': spec['md5'] if archive.exists() else None, 'official_files_checked': True}


def make_transforms(dataset='cifar10'):
    spec = dataset_spec(dataset)
    base = transforms.Compose([transforms.ToTensor(), transforms.Normalize(spec['mean'], spec['std'])])
    aug = transforms.Compose([transforms.RandomCrop(32, padding=4),
                              transforms.RandomHorizontalFlip(), base])
    return aug, base


def load(root, seed=42, smoke=False, dataset='cifar10'):
    prepare(root, dataset)
    spec = dataset_spec(dataset); count = spec['num_classes']
    aug, base = make_transforms(dataset)
    train = spec['constructor'](root, train=True, download=False, transform=aug)
    val = spec['constructor'](root, train=True, download=False, transform=base)
    labels = np.asarray(train.targets)
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for cls in range(count):
        indices = rng.permutation(np.flatnonzero(labels == cls))
        va.extend(indices[:5000 // count]); tr.extend(indices[5000 // count:])
    tr, va = np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64)
    assert len(tr) == 45000 and len(va) == 5000
    assert len(np.intersect1d(tr, va)) == 0
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(50000))
    assert np.all(np.bincount(labels[tr], minlength=count) == 45000 // count)
    assert np.all(np.bincount(labels[va], minlength=count) == 5000 // count)
    path = split_path(root, seed, dataset)
    if path.exists():
        with np.load(path) as old:
            if not (np.array_equal(old['train'], tr) and np.array_equal(old['val'], va)):
                raise ValueError(f'已有划分与 seed={seed} 不一致，拒绝覆盖 {path}')
    else:
        np.savez(path, train=tr, val=va)
    if smoke:
        tr = np.concatenate([tr[labels[tr] == c][:8] for c in range(count)])
        va = np.concatenate([va[labels[va] == c][:4] for c in range(count)])
    return Subset(train, tr.tolist()), Subset(val, va.tolist()), base


def testset(root, base, smoke=False, dataset='cifar10'):
    spec = dataset_spec(dataset); count = spec['num_classes']
    ds = spec['constructor'](root, train=False, download=False, transform=base)
    if smoke:
        labels = np.asarray(ds.targets)
        ids = np.concatenate([np.flatnonzero(labels == c)[:4] for c in range(count)])
        return Subset(ds, ids.tolist())
    return ds


def audit(root, seed=42, dataset='cifar10'):
    spec = dataset_spec(dataset); count = spec['num_classes']
    tr, va, base = load(root, seed, dataset=dataset)
    te = testset(root, base, dataset=dataset)
    labels = np.asarray(tr.dataset.targets)
    image1, label1 = va[0]; image2, label2 = va[0]
    assert np.array_equal(image1.numpy(), image2.numpy()) and label1 == label2
    assert tr.dataset is not va.dataset and tr.dataset.transform is not va.dataset.transform
    return {'source': f'official {dataset}', 'dataset': dataset, 'seed': seed,
            'train_size': len(tr), 'validation_size': len(va), 'test_size': len(te),
            'train_per_class': np.bincount(labels[tr.indices], minlength=count).tolist(),
            'validation_per_class': np.bincount(labels[va.indices], minlength=count).tolist(),
            'test_per_class': np.bincount(te.targets, minlength=count).tolist(),
            'disjoint_and_complete': True, 'validation_deterministic': True,
            'independent_transform_objects': True, 'split_md5': md5_file(split_path(root, seed, dataset)),
            'mean': spec['mean'], 'std': spec['std'], 'classes': tr.dataset.classes}
