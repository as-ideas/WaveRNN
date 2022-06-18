from pathlib import Path
from typing import Union, Callable, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from models.common_layers import CBHG, LengthRegulator
from utils.text.symbols import phonemes




class MelEncoder(nn.Module):

    def __init__(self, mel_dim, emb_dim=64, conv_dims=256, rnn_dims=64):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            BatchNormConv(mel_dim, conv_dims, 3, relu=True),
            BatchNormConv(conv_dims, conv_dims, 3, relu=True),
            BatchNormConv(conv_dims, conv_dims, 3, relu=True),
        ])
        self.rnn = nn.LSTM(conv_dims, rnn_dims, batch_first=True, bidirectional=False)
        self.lin = nn.Linear(2 * rnn_dims, 1)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        _, (x, _) = self.rnn(x)
        x = self.lin(x[-1])
        return x


class SeriesPredictor(nn.Module):

    def __init__(self, num_chars, emb_dim=64, conv_dims=256, rnn_dims=64, dropout=0.5, memb_dims=64):
        super().__init__()
        self.lang_embedding = nn.Embedding(2, 32)
        self.embedding = Embedding(num_chars, emb_dim)
        self.mel_encoder = MelEncoder(mel_dim=80, rnn_dims=memb_dims)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(emb_dim + 32, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims + memb_dims, 1)
        self.dropout = dropout

    def forward(self,
                x: torch.Tensor,
                lang_ind: torch.Tensor,
                mel: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        x = self.embedding(x)
        lemb = self.lang_embedding(lang_ind)
        memb = self.mel_encoder(mel)
        memb = memb[:, None, :]
        memb = memb.repeat(1, x.shape[1], 1)
        x = torch.cat([x, lemb], dim=2)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = torch.cat([x, memb], dim=-1)
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
                 padding_value=-11.5129):
        super().__init__()
        self.rnn_dims = rnn_dims
        self.padding_value = padding_value
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.lr = LengthRegulator()
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
                           in_channels=embed_dims + 32,
                           channels=prenet_dims,
                           proj_channels=[prenet_dims, embed_dims + 32],
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
        self.lang_embedding = nn.Embedding(num_embeddings=2, embedding_dim=32)
        self.pitch_proj = nn.Conv1d(1, 2 * prenet_dims + semb_dims, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv1d(1, 2 * prenet_dims + semb_dims, kernel_size=3, padding=1)
        self.mel_encoder = MelEncoder(mel_dim=n_mels, rnn_dims=256)


    def __repr__(self):
        num_params = sum([np.prod(p.size()) for p in self.parameters()])
        return f'ForwardTacotron, num params: {num_params}'

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch['x']
        mel = batch['mel']
        dur = batch['dur']
        mel_lens = batch['mel_len']
        pitch = batch['pitch'].unsqueeze(1)
        energy = batch['energy'].unsqueeze(1)
        lang_ind = torch.zeros(x.size(), dtype=torch.long, device=x.device)
        for b in range(x.size(0)):
            item_id = batch['item_id'][b]
            if str(item_id).startswith('p'):
                lang_ind[b, :] = 1

        if self.training:
            self.step += 1

        dur_hat = self.dur_pred(x, lang_ind, mel=mel).squeeze(-1)
        pitch_hat = self.pitch_pred(x, lang_ind, mel=mel).transpose(1, 2)
        energy_hat = self.energy_pred(x, lang_ind, mel=mel).transpose(1, 2)

        lemb = self.lang_embedding(lang_ind)
        x = self.embedding(x)
        x = torch.cat([x, lemb], dim=-1)
        x = x.transpose(1, 2)
        x = self.prenet(x)

        memb = self.mel_encoder(mel)
        memb = memb[:, None, :]
        memb = memb.repeat(1, x.shape[1], 1)
        x = torch.cat([x, memb], dim=2)

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
                'dur': dur_hat, 'pitch': pitch_hat, 'energy': energy_hat}

    def generate(self,
                 x: torch.Tensor,
                 lang_ind: int = 0,
                 mel: torch.Tensor = None,
                 alpha=1.0,
                 pitch_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                 energy_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            lang_inds = torch.full(x.size(), fill_value=lang_ind, device=x.device).long()
            dur_hat = self.dur_pred(x, lang_inds, mel, alpha=alpha)
            dur_hat = dur_hat.squeeze(2)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred(x, lang_inds, mel).transpose(1, 2)
            pitch_hat = pitch_function(pitch_hat)
            energy_hat = self.energy_pred(x, lang_inds, mel).transpose(1, 2)
            energy_hat = energy_function(energy_hat)
            return self._generate_mel(x=x, dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat,
                                      lang_inds=lang_inds,
                                      mel=mel)

    @torch.jit.export
    def generate_jit(self,
                     x: torch.Tensor,
                     alpha: float = 1.0,
                     beta: float = 1.0) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            dur_hat = self.dur_pred(x, alpha=alpha)
            dur_hat = dur_hat.squeeze(2)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred(x).transpose(1, 2) * beta
            energy_hat = self.energy_pred(x).transpose(1, 2)
            return self._generate_mel(x=x, dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat)

    def get_step(self) -> int:
        return self.step.data.item()

    def _generate_mel(self,
                      x: torch.Tensor,
                      lang_inds: torch.Tensor,
                      mel: torch.Tensor,
                      dur_hat: torch.Tensor,
                      pitch_hat: torch.Tensor,
                      energy_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
        lemb = self.lang_embedding(lang_inds)
        x = self.embedding(x)
        x = torch.cat([x, lemb], dim=-1)
        x = x.transpose(1, 2)
        x = self.prenet(x)
        memb = self.mel_encoder(mel)
        memb = memb[:, None, :]
        memb = memb.repeat(1, x.shape[1], 1)
        x = torch.cat([x, memb], dim=2)

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
                'pitch': pitch_hat, 'energy': energy_hat}

    def _pad(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', self.padding_value)
        return x


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ForwardTacotron':
        model_config = config['forward_tacotron']['model']
        model_config['num_chars'] = len(phonemes)
        model_config['n_mels'] = config['dsp']['num_mels']
        return ForwardTacotron(**model_config)

    @classmethod
    def from_checkpoint(cls, path: Union[Path, str]) -> 'ForwardTacotron':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = ForwardTacotron.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        return model