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


class SeriesPredictor(nn.Module):

    def __init__(self, num_chars, emb_dim=64, conv_dims=256,
                 rnn_dims=64, dropout=0.5, semb_dims=256, out_dims=512, p_dim=4):
        super().__init__()
        self.embedding = Embedding(num_chars, emb_dim)
        self.p_embedding = Embedding(out_dims, p_dim)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(emb_dim + semb_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(2*rnn_dims + p_dim, rnn_dims, batch_first=True, bidirectional=False)
        self.lin = nn.Linear(rnn_dims, out_dims)
        self.rnn_dims = rnn_dims
        self.dropout = dropout
        self.n_classes = out_dims

    def forward(self,
                x: torch.Tensor,
                semb: torch.Tensor,
                p_in: torch.Tensor) -> torch.Tensor:
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

        p_in = self.p_embedding(p_in.long())
        x_dec_in = torch.cat([x, p_in], dim=-1)
        x_out, _ = self.decoder(x_dec_in)
        x_out = self.lin(x_out)
        return x_out

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def generate(self, x, semb):
        x = self.embedding(x)
        #torch.fill_(semb, 0)
        speaker_emb = semb[:, None, :]
        speaker_emb = speaker_emb.repeat(1, x.shape[1], 1)
        x = torch.cat([x, speaker_emb], dim=2)
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)

        rnn = self.get_gru_cell(self.decoder)
        b_size, seq_len, _ = x.size()

        device = next(self.parameters()).device  # use same device as parameters
        h = torch.zeros(b_size, self.rnn_dims, device=device)
        o = torch.zeros(b_size, 1, device=device, dtype=torch.long)

        output = []

        with torch.no_grad():
            for i in range(seq_len):
                x_i = x[0, i:i+1, :]
                p_in = self.p_embedding(o.long()).squeeze(0)
                x_i = torch.cat([x_i, p_in], dim=-1)
                h = rnn(x_i, h)
                logits = self.lin(h)
                posterior = F.softmax(logits, dim=1)
                distrib = torch.distributions.Categorical(posterior)
                sample = distrib.sample().float()
                output.append(sample)
                o = sample.unsqueeze(0)

        output = torch.stack(output)
        return output


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
                 padding_value=-11.5129,
                 speaker_names: list = []):
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
        for speaker_name in speaker_names:
            self.register_buffer(speaker_name, torch.zeros(semb_dims, dtype=torch.float))

    def __repr__(self):
        num_params = sum([np.prod(p.size()) for p in self.parameters()])
        return f'ForwardTacotron, num params: {num_params}'

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch['x']
        mel = batch['mel']
        dur = batch['dur']
        semb = batch['speaker_emb']
        mel_lens = batch['mel_len']
        pitch = batch['pitch']
        pitch_target = batch['pitch_target']
        energy = batch['energy']

        device = next(self.parameters()).device  # use same device as parameters
        b, x_len = x.size()
        zeros = torch.zeros((b, 1), device=device)
        dur_in = torch.cat([zeros, dur[:, :-1]], dim=1)
        pitch_in = torch.cat([zeros, pitch_target[:, :-1]], dim=1)
        energy_in = torch.cat([zeros, energy[:, :-1]], dim=1)
        pitch_in = torch.clamp(pitch_in, max=511)
        energy_in = torch.clamp(energy_in, max=511)

        if self.training:
            self.step += 1

        dur_hat = self.dur_pred(x, semb, dur_in).squeeze(-1)
        pitch_hat = self.pitch_pred(x, semb, pitch_in).transpose(1, 2)
        energy_hat = self.energy_pred(x, semb, energy_in).transpose(1, 2)

        pitch = pitch.unsqueeze(1)
        energy = energy.unsqueeze(1)

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
                'dur': dur_hat, 'pitch': pitch_hat, 'energy': energy_hat}

    def generate(self,
                 x: torch.Tensor,
                 semb: torch.Tensor,
                 alpha=1.0,
                 pitch_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                 energy_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            dur_hat = self.dur_pred.generate(x, semb).unsqueeze(0).squeeze(2)
            print('gen dur')
            print(dur_hat.squeeze()[:10])
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred.generate(x, semb).unsqueeze(0).transpose(1, 2)
            pitch_hat = (pitch_hat - 256.) / 32.
            pitch_hat = pitch_function(pitch_hat)
            print('gen pitch')
            print(pitch_hat.squeeze()[:10])
            energy_hat = self.energy_pred.generate(x, semb).unsqueeze(0).transpose(1, 2)
            return self._generate_mel(x=x, dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat, semb=semb)

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
                      semb: torch.Tensor,
                      dur_hat: torch.Tensor,
                      pitch_hat: torch.Tensor,
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