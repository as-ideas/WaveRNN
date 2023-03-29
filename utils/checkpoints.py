from pathlib import Path
from typing import Dict, Any, Union

import torch
import torch.optim.optimizer
from models.fast_pitch import FastPitch
from models.forward_tacotron import ForwardTacotron
from models.multi_fast_pitch import MultiFastPitch
from models.multi_forward_tacotron import MultiForwardTacotron
from models.tacotron import Tacotron


def save_checkpoint(model: torch.nn.Module,
                    optim: torch.optim.Optimizer,
                    config: Dict[str, Any],
                    path: Path,
                    meta: Dict[str, Any] = None) -> None:
    checkpoint = {'model': model.state_dict(),
                  'optim': optim.state_dict(),
                  'config': config}
    if meta is not None:
        checkpoint.update(meta)
    torch.save(checkpoint, str(path))


def restore_checkpoint(model: Union[FastPitch, ForwardTacotron, Tacotron, MultiForwardTacotron, MultiFastPitch],
                       optim: torch.optim.Optimizer,
                       path: Path,
                       device: torch.device) -> None:
    if path.is_file():
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        print(f'Restored model with step {model.get_step()}\n')


def init_tts_model(config: Dict[str, Any]) -> Union[ForwardTacotron, FastPitch, MultiForwardTacotron, MultiFastPitch]:
    model_type = config.get('tts_model', 'forward_tacotron')
    if model_type == 'forward_tacotron':
        model = ForwardTacotron.from_config(config)
    elif model_type == 'fast_pitch':
        model = FastPitch.from_config(config)
    elif model_type == 'multi_forward_tacotron':
        model = MultiForwardTacotron.from_config(config)
    elif model_type == 'multi_fast_pitch':
        model = MultiFastPitch.from_config(config)
    else:
        raise ValueError(f'Model type not supported: {model_type}')
    return model
