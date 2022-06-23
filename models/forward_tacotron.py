from pathlib import Path
from typing import Union, Callable, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Conv1d, LayerNorm, ReLU, Linear
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from models.common_layers import CBHG, LengthRegulator
from utils.text.symbols import phonemes


class SeriesPredictor(nn.Module):

    def __init__(self, num_chars, emb_dim=64, conv_dims=256, rnn_dims=64, dropout=0.5, semb_dims=256, out_dim=1):
        super().__init__()
        self.embedding = Embedding(num_chars, emb_dim)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(emb_dim + semb_dims + 64, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, out_dim)
        self.dropout = dropout

    def forward(self,
                x: torch.Tensor,
                semb: torch.Tensor,
                ada: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        x = self.embedding(x)
        speaker_emb = semb[:, None, :]
        speaker_emb = speaker_emb.repeat(1, x.shape[1], 1)
        x = torch.cat([x, speaker_emb, ada], dim=2)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x / alpha


class BatchNormConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel: int, relu: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.relu:
            x = F.relu(x)
        x = self.bnorm(x)
        return x


class PhonPredictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = BatchNormConv(80, 256, 3, relu=True)
        self.conv2 = BatchNormConv(256, 256, 3, relu=True)
        self.lin = Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.lin(x)
        return x


class ForwardTacotron(nn.Module):

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
                 semb_dims: int = 256,
                 padding_value=-11.5129,
                 speaker_names: list = []):
        super().__init__()
        self.rnn_dims = rnn_dims
        self.padding_value = padding_value
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()

        self.phon_pred = SeriesPredictor(num_chars=num_chars,
                                         emb_dim=series_embed_dims,
                                         conv_dims=pitch_conv_dims,
                                         rnn_dims=pitch_rnn_dims,
                                         dropout=pitch_dropout,
                                         out_dim=4)

        self.phon_train_pred = PhonPredictor()
        self.phon_series_lin = Linear(4, 64)

        self.dur_pred = SeriesPredictor(num_chars=num_chars,
                                        emb_dim=series_embed_dims,
                                        conv_dims=durpred_conv_dims,
                                        rnn_dims=durpred_rnn_dims,
                                        dropout=durpred_dropout)
        self.pitch_pred = SeriesPredictor(num_chars=num_chars,
                                          emb_dim=series_embed_dims,
                                          conv_dims=pitch_conv_dims,
                                          rnn_dims=pitch_rnn_dims,
                                          dropout=pitch_dropout)
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
        self.lstm = nn.LSTM(2 * prenet_dims + semb_dims,
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
        self.pitch_proj = nn.Conv1d(1, 2 * prenet_dims + semb_dims, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv1d(1, 2 * prenet_dims + semb_dims, kernel_size=3, padding=1)
        self.ada_proj = nn.Conv1d(4, 2 * prenet_dims + semb_dims, kernel_size=3, padding=1)
        for speaker_name in speaker_names:
            self.register_buffer(speaker_name, torch.zeros(semb_dims, dtype=torch.float))

    def __repr__(self):
        num_params = sum([np.prod(p.size()) for p in self.parameters()])
        return f'ForwardTacotron, num params: {num_params}'

    def forward(self, batch: Dict[str, torch.Tensor], train=True) -> Dict[str, torch.Tensor]:
        x = batch['x']
        mel = batch['mel']
        dur = batch['dur']
        semb = batch['speaker_emb']
        mel_lens = batch['mel_len']
        pitch = batch['pitch'].unsqueeze(1)
        energy = batch['energy'].unsqueeze(1)

        if self.training:
            self.step += 1


        B, T = x.size()
        ada_in = torch.zeros((B, 80, T), device=x.device)
        for b in range(B):
            t1 = 0
            for t in range(T):
                t2 = t1 + int(dur[b, t])
                ada_in[b, :, t] = mel[b, :, t1:t2].mean(dim=1)
                t1 = t2
        ada_in[ada_in != ada_in] = 0
        ada_target = self.phon_train_pred(ada_in)

        ada_hat = self.phon_pred(x, semb)

        ada_target_in = ada_target if train else ada_hat

        ada_series = self.phon_series_lin(ada_target_in)

        dur_hat = self.dur_pred(x, semb, ada_series).squeeze(-1)
        pitch_hat = self.pitch_pred(x, semb, ada_series).transpose(1, 2)
        energy_hat = self.energy_pred(x, semb, ada_series).transpose(1, 2)

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

        ada_proj = self.ada_proj(ada_target)
        ada_proj = ada_proj.transpose(1, 2)
        x = x + ada_proj * self.ada_strength

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
                'dur': dur_hat, 'pitch': pitch_hat, 'energy': energy_hat,
                'ada_hat': ada_hat, 'ada_target': ada_target}

    def generate(self,
                 x: torch.Tensor,
                 semb: torch.Tensor,
                 alpha=1.0,
                 pitch_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                 energy_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            ada_hat = self.phon_pred(x, semb)
            ada_series = self.phon_series_lin(ada_hat)

            dur_hat = self.dur_pred(x, semb, ada_series, alpha=alpha)
            dur_hat = dur_hat.squeeze(2)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred(x, semb, ada_series).transpose(1, 2)
            pitch_hat = pitch_function(pitch_hat)
            energy_hat = self.energy_pred(x, semb, ada_series).transpose(1, 2)
            energy_hat = energy_function(energy_hat)

        return self._generate_mel(x=x, dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat, semb=semb, ada_hat=ada_hat)

    def get_step(self) -> int:
        return self.step.data.item()

    def _generate_mel(self,
                      x: torch.Tensor,
                      semb: torch.Tensor,
                      dur_hat: torch.Tensor,
                      pitch_hat: torch.Tensor,
                      ada_hat: torch.Tensor,
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

        ada_proj = self.ada_proj(ada_hat)
        ada_proj = ada_proj.transpose(1, 2)
        x = x + ada_proj * self.ada_strength

        x = self.lr(x, dur_hat)

        x, _ = self.lstm(x)

        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        return {'mel': x, 'mel_post': x_post, 'dur': dur_hat,
                'pitch': pitch_hat, 'energy': energy_hat}

    def _pad(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', self.padding_value)
        return x


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ForwardTacotron':
        speaker_names = config['speaker_names']
        model_config = config['forward_tacotron']['model']
        model_config['num_chars'] = len(phonemes)
        model_config['n_mels'] = config['dsp']['num_mels']
        model_config['speaker_names'] = speaker_names
        return ForwardTacotron(**model_config)

    @classmethod
    def from_checkpoint(cls, path: Union[Path, str]) -> 'ForwardTacotron':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = ForwardTacotron.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        return model