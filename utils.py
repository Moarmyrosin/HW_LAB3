"""配置、随机状态与运行环境记录。"""
import json
import os
import platform
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda'] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def environment(device):
    import torchvision
    import matplotlib
    import sklearn
    return {'recorded_at': datetime.now().astimezone().isoformat(),
            'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
            'sklearn': sklearn.__version__, 'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'gpu_memory_bytes': torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None,
            'deterministic_cudnn': True, 'tf32': False,
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'limits': '固定种子不保证跨硬件、跨库版本逐位一致；多 worker 数据增强和 AMP 也可能产生差异。'}


def choose_device(name='auto'):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('请求 CUDA，但当前不可用')
    return device


def checkpoint_save(path, state):
    path = Path(path); temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary); temporary.replace(path)
