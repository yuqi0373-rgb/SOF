import json
from pathlib import Path

import torch

from .model import build_model


def save_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: torch.zeros_like(value, device='cpu') if name.endswith('num_batches_tracked')
        else value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_model(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError('Expected a tensor state dictionary')
    if 'conv1.conv.0.weight' in state:
        backbone = 'rolling'
    elif 'encoder1.conv.0.weight' in state:
        backbone = 'ukan'
    else:
        raise ValueError('Unrecognized model weights')
    model = build_model(backbone, num_classes=state['final.weight'].shape[0])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def recommended_size(path):
    root = Path(__file__).resolve().parents[1] / 'weights'
    try:
        name = str(Path(path).resolve().relative_to(root))
        index = json.loads((root / 'index.json').read_text())
        return next(item['input_size'] for item in index['weights'] if item['path'] == name)
    except (ValueError, FileNotFoundError, StopIteration):
        return 256
