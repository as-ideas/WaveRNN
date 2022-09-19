import copy
import math
from pathlib import Path
from typing import Union, Callable, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, LayerNorm, MultiheadAttention

from models.common_layers import LengthRegulator
from utils.text.symbols import phonemes


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout=0.1, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale = torch.nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # shape: [T, N]
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def make_token_len_mask(x: torch.Tensor) -> torch.Tensor:
    return (x == 0).transpose(0, 1)


def make_mel_len_mask(x: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
    len_mask = torch.zeros((x.size(0), x.size(1))).bool().to(x.device)
    for i, mel_len in enumerate(mel_lens):
        len_mask[i, mel_len:] = True
    return len_mask


class FFTBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 conv1_kernel: int,
                 conv2_kernel: int,
                 d_fft: int,
                 dropout: float = 0.1):
        super(FFTBlock, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_fft,
                               kernel_size=conv1_kernel, stride=1, padding=conv1_kernel//2)
        self.conv2 = nn.Conv1d(in_channels=d_fft, out_channels=d_model,
                               kernel_size=conv2_kernel, stride=1, padding=conv2_kernel//2)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()

    def forward(self,
                src: torch.Tensor,
                src_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        src2 = self.self_attn(src, src, src,
                              attn_mask=None,
                              key_padding_mask=src_pad_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src.transpose(0, 1).transpose(1, 2)
        src2 = self.conv1(src)
        src2 = self.activation(src2)
        src2 = self.conv2(src2)
        src = src + self.dropout2(src2)
        src = src.transpose(1, 2).transpose(0, 1)
        src = self.norm2(src)
        return src


class ForwardTransformer(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 d_fft: int,
                 layers: int,
                 heads: int,
                 conv1_kernel: int,
                 conv2_kernel: int,
                 dropout: float = 0.1,
                 ) -> None:
        super().__init__()

        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = FFTBlock(d_model=d_model,
                                 nhead=heads,
                                 d_fft=d_fft,
                                 conv1_kernel=conv1_kernel,
                                 conv2_kernel=conv2_kernel,
                                 dropout=dropout)
        encoder_norm = LayerNorm(d_model)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer)
                                     for _ in range(layers)])
        self.norm = encoder_norm

    def forward(self,
                x: torch.Tensor,
                src_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:         # shape: [N, T]
        x = x.transpose(0, 1)        # shape: [T, N]
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_pad_mask=src_pad_mask)
        x = self.norm(x)
        x = x.transpose(0, 1)
        return x


class SeriesPredictor(nn.Module):

    def __init__(self,
                 num_chars: int,
                 d_model: int,
                 n_heads: int,
                 d_fft: int,
                 layers: int,
                 conv1_kernel: int,
                 conv2_kernel: int,
                 semb_dim: int = 256,
                 dropout=0.1):
        super().__init__()
        self.embedding = Embedding(num_chars, d_model)
        self.transformer = ForwardTransformer(heads=n_heads, dropout=dropout,
                                              d_model=d_model+semb_dim, d_fft=d_fft,
                                              conv1_kernel=conv1_kernel,
                                              conv2_kernel=conv2_kernel,
                                              layers=layers)
        self.lin = nn.Linear(d_model+semb_dim, 1)

    def forward(self,
                x: torch.Tensor,
                semb: torch.Tensor,
                src_pad_mask: Optional[torch.Tensor] = None,
                alpha: float = 1.0) -> torch.Tensor:
        x = self.embedding(x)
        speaker_emb = semb[:, None, :]
        speaker_emb = speaker_emb.repeat(1, x.shape[1], 1)
        x = torch.cat([x, speaker_emb], dim=2)
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
                 semb_dims: int = 256,
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
                                          d_model=d_model+semb_dims, d_fft=postnet_fft, layers=postnet_layers)
        self.lin = torch.nn.Linear(d_model+semb_dims, n_mels)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.pitch_strength = pitch_strength
        self.energy_strength = energy_strength
        self.pitch_proj = nn.Conv1d(1, d_model + semb_dims, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv1d(1, d_model + semb_dims, kernel_size=3, padding=1)

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
        semb = batch['speaker_emb']

        if self.training:
            self.step += 1

        len_mask = make_token_len_mask(x.transpose(0, 1))
        dur_hat = self.dur_pred(x, semb, src_pad_mask=len_mask).squeeze(-1)
        pitch_hat = self.pitch_pred(x, semb, src_pad_mask=len_mask).transpose(1, 2)
        energy_hat = self.energy_pred(x, semb, src_pad_mask=len_mask).transpose(1, 2)

        x = self.embedding(x)
        x = self.prenet(x, src_pad_mask=len_mask)

        speaker_emb = semb[:, None, :]
        speaker_emb = speaker_emb.repeat(1, x.shape[1], 1)
        x = torch.cat([x, speaker_emb], dim=2)

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
                 semb: torch.Tensor,
                 alpha=1.0,
                 pitch_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                 energy_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            dur_hat = self.dur_pred(x, semb, alpha=alpha)
            dur_hat = dur_hat.squeeze(2)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred(x, semb).transpose(1, 2)
            pitch_hat = pitch_function(pitch_hat)
            energy_hat = self.energy_pred(x, semb).transpose(1, 2)
            energy_hat = energy_function(energy_hat)
            return self._generate_mel(x=x, dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat, semb=semb)

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
                      energy_hat: torch.Tensor,
                      semb: torch.Tensor) -> Dict[str, torch.Tensor]:

        len_mask = make_token_len_mask(x.transpose(0, 1))

        x = self.embedding(x)
        x = self.prenet(x, src_pad_mask=len_mask)
        speaker_emb = semb[:, None, :]
        speaker_emb = speaker_emb.repeat(1, x.shape[1], 1)
        x = torch.cat([x, speaker_emb], dim=2)

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