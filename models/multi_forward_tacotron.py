from pathlib import Path
from typing import Union, Callable, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.common_layers import CBHG, LengthRegulator, BatchNormConv
from utils.text.symbols import phonemes


class SeriesPredictor(nn.Module):

    def __init__(self,
                 num_chars: int,
                 emb_dim: int = 64,
                 conv_dims: int = 256,
                 rnn_dims: int = 64,
                 dropout: float = 0.5,
                 speaker_emb_dims: int = 256,
                 out_dim: int = 1):
        super().__init__()
        self.embedding = Embedding(num_chars, emb_dim)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(emb_dim + speaker_emb_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, out_dim)
        self.dropout = dropout

    def forward(self,
                x: torch.Tensor,
                semb: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        x = self.embedding(x)
        speaker_emb = semb[:, None, :]
        speaker_emb = speaker_emb.repeat(1, x.shape[1], 1)
        x = torch.cat([x, speaker_emb], dim=2)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x / alpha


class ConditionalSeriesPredictor(nn.Module):

    def __init__(self,
                 num_chars: int,
                 emb_dim: int = 64,
                 cond_emb_size: int = 4,
                 cond_emb_dims: int = 8,
                 conv_dims: int = 256,
                 rnn_dims: int = 64,
                 dropout: float = 0.5,
                 speaker_emb_dims: int = 256):
        super().__init__()
        self.embedding = Embedding(num_chars, emb_dim)
        self.pitch_cond_embedding = Embedding(cond_emb_size, cond_emb_dims)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(emb_dim + cond_emb_dims + speaker_emb_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, 1)
        self.dropout = dropout

    def forward(self,
                x: torch.Tensor,
                x_cond: torch.Tensor,
                speaker_emb: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        x = self.embedding(x)
        x_cond = self.pitch_cond_embedding(x_cond)
        speaker_emb = speaker_emb[:, None, :]
        speaker_emb = speaker_emb.repeat(1, x.shape[1], 1)
        x = torch.cat([x, x_cond, speaker_emb], dim=2)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x / alpha


class MultiForwardTacotron(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 series_embed_dims: int,
                 num_chars: int,
                 durpred_conv_dims: int,
                 durpred_rnn_dims: int,
                 durpred_dropout: float,
                 pitch_conv_dims: int,
                 pitch_rnn_dims: int,
                 pitch_dropout: float,
                 pitch_strength: float,
                 pitch_cond_conv_dims: int,
                 pitch_cond_rnn_dims: int,
                 pitch_cond_dropout: float,
                 energy_conv_dims: int,
                 energy_rnn_dims: int,
                 energy_dropout: float,
                 energy_strength: float,
                 rnn_dims: int,
                 prenet_dims: int,
                 prenet_k: int,
                 postnet_num_highways: int,
                 prenet_dropout: float,
                 postnet_dims: int,
                 postnet_k: int,
                 prenet_num_highways: int,
                 postnet_dropout: float,
                 n_mels: int,
                 speaker_emb_dims: int,
                 pitch_cond_emb_dims: int,
                 pitch_cond_categorical_dims: int,
                 padding_value=-11.5129):
        super().__init__()
        self.rnn_dims = rnn_dims
        self.padding_value = padding_value
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()
        self.dur_pred = ConditionalSeriesPredictor(num_chars=num_chars,
                                                   emb_dim=series_embed_dims,
                                                   conv_dims=durpred_conv_dims,
                                                   rnn_dims=durpred_rnn_dims,
                                                   cond_emb_dims=pitch_cond_emb_dims,
                                                   dropout=durpred_dropout)
        self.pitch_cond_pred = SeriesPredictor(num_chars=num_chars,
                                               emb_dim=series_embed_dims,
                                               conv_dims=pitch_cond_conv_dims,
                                               rnn_dims=pitch_cond_rnn_dims,
                                               dropout=pitch_cond_dropout,
                                               out_dim=pitch_cond_categorical_dims)
        self.pitch_pred = ConditionalSeriesPredictor(num_chars=num_chars,
                                                     emb_dim=series_embed_dims,
                                                     conv_dims=pitch_conv_dims,
                                                     rnn_dims=pitch_rnn_dims,
                                                     cond_emb_dims=pitch_cond_emb_dims,
                                                     dropout=pitch_dropout, )
        self.energy_pred = SeriesPredictor(num_chars=num_chars,
                                           emb_dim=series_embed_dims,
                                           conv_dims=energy_conv_dims,
                                           rnn_dims=energy_rnn_dims,
                                           dropout=energy_dropout)
        self.prenet = CBHG(K=prenet_k,
                           in_channels=embed_dims,
                           channels=prenet_dims,
                           proj_channels=[prenet_dims, embed_dims],
                           num_highways=prenet_num_highways,
                           dropout=prenet_dropout)
        self.lstm = nn.LSTM(2 * prenet_dims + speaker_emb_dims,
                            rnn_dims,
                            batch_first=True,
                            bidirectional=True)
        self.lin = torch.nn.Linear(2 * rnn_dims, n_mels)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.postnet = CBHG(K=postnet_k,
                            in_channels=n_mels,
                            channels=postnet_dims,
                            proj_channels=[postnet_dims, n_mels],
                            num_highways=postnet_num_highways,
                            dropout=postnet_dropout)
        self.post_proj = nn.Linear(2 * postnet_dims, n_mels, bias=False)
        self.pitch_strength = pitch_strength
        self.energy_strength = energy_strength
        self.pitch_proj = nn.Conv1d(1, 2 * prenet_dims + speaker_emb_dims, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv1d(1, 2 * prenet_dims + speaker_emb_dims, kernel_size=3, padding=1)

    def __repr__(self):
        num_params = sum([np.prod(p.size()) for p in self.parameters()])
        return f'MultiForwardTacotron, num params: {num_params}'

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch['x']
        mel = batch['mel']
        dur = batch['dur']
        semb = batch['speaker_emb']
        mel_lens = batch['mel_len']
        pitch = batch['pitch'].unsqueeze(1)
        pitch_cond = batch['pitch_cond']
        energy = batch['energy'].unsqueeze(1)

        if self.training:
            self.step += 1

        pitch_cond_hat = self.pitch_cond_pred(x, semb).squeeze(-1)

        dur_hat = self.dur_pred(x, pitch_cond, semb).squeeze(-1)
        pitch_hat = self.pitch_pred(x, pitch_cond, semb).transpose(1, 2)
        energy_hat = self.energy_pred(x, semb).transpose(1, 2)

        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.prenet(x)
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

        x = pack_padded_sequence(x, lengths=mel_lens.cpu(), enforce_sorted=False,
                                 batch_first=True)

        x, _ = self.lstm(x)

        x, _ = pad_packed_sequence(x, padding_value=self.padding_value, batch_first=True)

        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        x_post = self._pad(x_post, mel.size(2))
        x = self._pad(x, mel.size(2))

        return {'mel': x, 'mel_post': x_post,
                'dur': dur_hat, 'pitch': pitch_hat,
                'energy': energy_hat, 'pitch_cond': pitch_cond_hat}

    def generate(self,
                 x: torch.Tensor,
                 speaker_emb: torch.Tensor,
                 alpha=1.0,
                 pitch_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                 energy_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            pitch_cond_hat = self.pitch_cond_pred(x, speaker_emb).squeeze(-1)
            pitch_cond_hat = torch.argmax(pitch_cond_hat.squeeze(), dim=1).long().unsqueeze(0)
            dur_hat = self.dur_pred(x, pitch_cond_hat, speaker_emb, alpha=alpha).squeeze(-1)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred(x, pitch_cond_hat, speaker_emb).transpose(1, 2)
            pitch_hat = pitch_function(pitch_hat)
            energy_hat = self.energy_pred(x, speaker_emb).transpose(1, 2)
            energy_hat = energy_function(energy_hat)
            return self._generate_mel(x=x,
                                      dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat,
                                      pitch_cond_hat=pitch_cond_hat,
                                      semb=speaker_emb)

    def get_step(self) -> int:
        return self.step.data.item()

    def _generate_mel(self,
                      x: torch.Tensor,
                      semb: torch.Tensor,
                      dur_hat: torch.Tensor,
                      pitch_hat: torch.Tensor,
                      pitch_cond_hat: torch,
                      energy_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.prenet(x)
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

        x, _ = self.lstm(x)

        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        return {'mel': x, 'mel_post': x_post, 'dur': dur_hat,
                'pitch': pitch_hat, 'energy': energy_hat,
                'pitch_cond': pitch_cond_hat.unsqueeze(1)}

    def _pad(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', self.padding_value)
        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MultiForwardTacotron':
        model_config = config['multi_forward_tacotron']['model']
        model_config['num_chars'] = len(phonemes)
        model_config['n_mels'] = config['dsp']['num_mels']
        return MultiForwardTacotron(**model_config)

    @classmethod
    def from_checkpoint(cls, path: Union[Path, str]) -> 'MultiForwardTacotron':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = MultiForwardTacotron.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        return model