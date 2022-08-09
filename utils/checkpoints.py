from pathlib import Path
from typing import Tuple, Dict, Any, Union

import torch
import torch.optim.optimizer
from models.deepmind_version import WaveRNN
from models.fast_pitch import FastPitch
from models.forward_tacotron import ForwardTacotron
from models.tacotron import Tacotron


def save_checkpoint(model: torch.nn.Module,
                    optim: torch.optim.Optimizer,
                    config: Dict[str, Any],
                    path: Path) -> None:
    torch.save({'model': model.state_dict(),
                'optim': optim.state_dict(),
                'config': config}, str(path))


def restore_checkpoint(model: Union[FastPitch, ForwardTacotron, Tacotron, WaveRNN],
                       optim: torch.optim.Optimizer,
                       path: Path,
                       device: torch.device) -> None:
    if path.is_file():
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        optim.load_state_dict(checkpoint['optim'])
        print(f'Restored model with step {model.get_step()}\n')


def init_tts_model(config: Dict[str, Any]) -> Union[ForwardTacotron, FastPitch]:
    model_type = config.get('tts_model', 'forward_tacotron')
    if model_type == 'forward_tacotron':
        model = ForwardTacotron.from_config(config)
    elif model_type == 'fast_pitch':
        model = FastPitch.from_config(config)
    else:
        raise ValueError(f'Model type not supported: {model_type}')
    return model
