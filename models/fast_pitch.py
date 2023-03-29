from pathlib import Path
from typing import Union, Callable, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding

from models.common_layers import LengthRegulator, ForwardTransformer, make_token_len_mask
from utils.text.symbols import phonemes


class SeriesPredictor(nn.Module):

    def __init__(self,
                 num_chars: int,
                 d_model: int,
                 n_heads: int,
                 d_fft: int,
                 layers: int,
                 conv1_kernel: int,
                 conv2_kernel: int,
                 dropout=0.1):
        super().__init__()
        self.embedding = Embedding(num_chars, d_model)
        self.transformer = ForwardTransformer(heads=n_heads, dropout=dropout,
                                              d_model=d_model, d_fft=d_fft,
                                              conv1_kernel=conv1_kernel,
                                              conv2_kernel=conv2_kernel,
                                              layers=layers)
        self.lin = nn.Linear(d_model, 1)

    def forward(self,
                x: torch.Tensor,
                src_pad_mask: Optional[torch.Tensor] = None,
                alpha: float = 1.0) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x, src_pad_mask=src_pad_mask)
        x = self.lin(x)
        return x / alpha


class FastPitch(nn.Module):

    def __init__(self,
                 num_chars: int,
                 durpred_dropout: float,
                 durpred_d_model: int,
                 durpred_n_heads: int,
                 durpred_layers: int,
                 durpred_d_fft: int,
                 pitch_dropout: float,
                 pitch_d_model: int,
                 pitch_n_heads: int,
                 pitch_layers: int,
                 pitch_d_fft: int,
                 energy_dropout: float,
                 energy_d_model: int,
                 energy_n_heads: int,
                 energy_layers: int,
                 energy_d_fft: int,
                 pitch_strength: float,
                 energy_strength: float,
                 d_model: int,
                 conv1_kernel: int,
                 conv2_kernel: int,
                 prenet_layers: int,
                 prenet_heads: int,
                 prenet_fft: int,
                 prenet_dropout: float,
                 postnet_layers: int,
                 postnet_heads: int,
                 postnet_fft: int,
                 postnet_dropout: float,
                 n_mels: int,
                 padding_value=-11.5129):
        super().__init__()
        self.padding_value = padding_value
        self.lr = LengthRegulator()
        self.dur_pred = SeriesPredictor(num_chars=num_chars,
                                        d_model=durpred_d_model,
                                        n_heads=durpred_n_heads,
                                        layers=durpred_layers,
                                        d_fft=durpred_d_fft,
                                        conv1_kernel=conv1_kernel,
                                        conv2_kernel=conv2_kernel,
                                        dropout=durpred_dropout)
        self.pitch_pred = SeriesPredictor(num_chars=num_chars,
                                          d_model=pitch_d_model,
                                          n_heads=pitch_n_heads,
                                          layers=pitch_layers,
                                          d_fft=pitch_d_fft,
                                          conv1_kernel=conv1_kernel,
                                          conv2_kernel=conv2_kernel,
                                          dropout=pitch_dropout)
        self.energy_pred = SeriesPredictor(num_chars=num_chars,
                                           d_model=energy_d_model,
                                           n_heads=energy_n_heads,
                                           layers=energy_layers,
                                           d_fft=energy_d_fft,
                                           conv1_kernel=conv1_kernel,
                                           conv2_kernel=conv2_kernel,
                                           dropout=energy_dropout)
        self.embedding = Embedding(num_embeddings=num_chars, embedding_dim=d_model)
        self.prenet = ForwardTransformer(heads=prenet_heads, dropout=prenet_dropout,
                                         conv1_kernel=conv1_kernel, conv2_kernel=conv2_kernel,
                                         d_model=d_model, d_fft=prenet_fft, layers=prenet_layers)
        self.postnet = ForwardTransformer(heads=postnet_heads, dropout=postnet_dropout,
                                          conv1_kernel=conv1_kernel, conv2_kernel=conv2_kernel,
                                          d_model=d_model, d_fft=postnet_fft, layers=postnet_layers)
        self.lin = torch.nn.Linear(d_model, n_mels)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.pitch_strength = pitch_strength
        self.energy_strength = energy_strength
        self.pitch_proj = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv1d(1, d_model, kernel_size=3, padding=1)

    def __repr__(self):
        num_params = sum([np.prod(p.size()) for p in self.parameters()])
        return f'FastPitch, num params: {num_params}'

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch['x']
        mel = batch['mel']
        dur = batch['dur']
        mel_lens = batch['mel_len']
        pitch = batch['pitch'].unsqueeze(1)
        energy = batch['energy'].unsqueeze(1)

        if self.training:
            self.step += 1

        len_mask = make_token_len_mask(x.transpose(0, 1))
        dur_hat = self.dur_pred(x, src_pad_mask=len_mask).squeeze(-1)
        pitch_hat = self.pitch_pred(x, src_pad_mask=len_mask).transpose(1, 2)
        energy_hat = self.energy_pred(x, src_pad_mask=len_mask).transpose(1, 2)

        x = self.embedding(x)
        x = self.prenet(x, src_pad_mask=len_mask)

        pitch_proj = self.pitch_proj(pitch)
        pitch_proj = pitch_proj.transpose(1, 2)
        x = x + pitch_proj * self.pitch_strength

        energy_proj = self.energy_proj(energy)
        energy_proj = energy_proj.transpose(1, 2)
        x = x + energy_proj * self.energy_strength

        x = self.lr(x, dur)

        len_mask = torch.zeros((x.size(0), x.size(1))).bool().to(x.device)
        for i, mel_len in enumerate(mel_lens):
            len_mask[i, mel_len:] = True

        x = self.postnet(x, src_pad_mask=len_mask)

        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.pad(x, mel.size(2))
        x = self.pad(x, mel.size(2))

        return {'mel': x, 'mel_post': x_post,
                'dur': dur_hat, 'pitch': pitch_hat, 'energy': energy_hat}

    def generate(self,
                 x: torch.Tensor,
                 alpha=1.0,
                 pitch_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                 energy_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            dur_hat = self.dur_pred(x, alpha=alpha)
            dur_hat = dur_hat.squeeze(2)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred(x).transpose(1, 2)
            pitch_hat = pitch_function(pitch_hat)
            energy_hat = self.energy_pred(x).transpose(1, 2)
            energy_hat = energy_function(energy_hat)
            return self._generate_mel(x=x, dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat)

    def pad(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', self.padding_value)
        return x

    def get_step(self) -> int:
        return self.step.data.item()

    def _generate_mel(self,
                      x: torch.Tensor,
                      dur_hat: torch.Tensor,
                      pitch_hat: torch.Tensor,
                      energy_hat: torch.Tensor) -> Dict[str, torch.Tensor]:

        len_mask = make_token_len_mask(x.transpose(0, 1))

        x = self.embedding(x)
        x = self.prenet(x, src_pad_mask=len_mask)

        pitch_proj = self.pitch_proj(pitch_hat)
        pitch_proj = pitch_proj.transpose(1, 2)
        x = x + pitch_proj * self.pitch_strength

        energy_proj = self.energy_proj(energy_hat)
        energy_proj = energy_proj.transpose(1, 2)
        x = x + energy_proj * self.energy_strength

        x = self.lr(x, dur_hat)

        x = self.postnet(x, src_pad_mask=None)

        x = self.lin(x)
        x = x.transpose(1, 2)

        return {'mel': x, 'mel_post': x, 'dur': dur_hat,
                'pitch': pitch_hat, 'energy': energy_hat}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FastPitch':
        model_config = config['fast_pitch']['model']
        model_config['num_chars'] = len(phonemes)
        model_config['n_mels'] = config['dsp']['num_mels']
        return FastPitch(**model_config)

    @classmethod
    def from_checkpoint(cls, path: Union[Path, str]) -> 'FastPitch':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = FastPitch.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        return model
