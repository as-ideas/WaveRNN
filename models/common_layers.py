import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, MultiheadAttention
#from torch.nn.utils.rnn import pad_sequence

class CustomPadSequence(nn.Module):
    def __init__(self, padding_value=0.):
        super().__init__()
        self.padding_value = padding_value

    def forward(self, sequences):
        # Find the maximum length in the sequences
        max_length = max(len(seq) for seq in sequences)

        # Pad sequences with the specified padding value
        padded_sequences = [F.pad(seq, (0, max_length - len(seq)), value=self.padding_value) for seq in sequences]

        # Stack the padded sequences
        padded_sequences = torch.stack(padded_sequences, dim=0)

        return padded_sequences

class LengthRegulator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dur: torch.Tensor) -> torch.Tensor:
        dur[dur < 0] = 0.
        x_expanded = []
        for i in range(x.size(0)):
            x_exp = torch.repeat_interleave(x[i], (dur[i] + 0.5).long(), dim=0)
            x_expanded.append(x_exp)
        x_expanded = pad_sequence(x_expanded, padding_value=0., batch_first=True)
        return x_expanded

class LengthRegulator_onnx(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dur: torch.Tensor) -> torch.Tensor:
        dur[dur < 0] = 0.
        x_expanded = []
        for i in range(x.size(0)):
            x_exp = torch.repeat_interleave(x[i], (dur[i] + 0.5).long(), dim=0)
            x_expanded.append(x_exp)
        customPadSequence = CustomPadSequence(padding_value=0.)
        x_expanded = customPadSequence(x_expanded)
        return x_expanded



class HighwayNetwork(nn.Module):

    def __init__(self, size: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)
        self.W1.bias.data.fill_(0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.W1(x)
        x2 = self.W2(x)
        g = torch.sigmoid(x2)
        y = g * F.relu(x1) + (1. - g) * x
        return y


class BatchNormConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: int, relu=True) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x) if self.relu is True else x
        return self.bnorm(x)


class CBHG(nn.Module):

    def __init__(self,
                 K: int,
                 in_channels: int,
                 channels: int,
                 proj_channels: list,
                 num_highways: int,
                 dropout: float = 0.5) -> None:
        super().__init__()

        self.dropout = dropout
        self.bank_kernels = [i for i in range(1, K + 1)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            conv = BatchNormConv(in_channels, channels, k)
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        self.conv_project1 = BatchNormConv(len(self.bank_kernels) * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)

        self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        self.highways = nn.ModuleList()
        for i in range(num_highways):
            hn = HighwayNetwork(channels)
            self.highways.append(hn)

        self.rnn = nn.GRU(channels, channels, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        seq_len = x.size(-1)
        conv_bank = []

        # Convolution Bank
        for conv in self.conv1d_bank:
            c = conv(x)  # Convolution
            conv_bank.append(c[:, :, :seq_len])

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)[:, :, :seq_len]
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Conv1d projections
        x = self.conv_project1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        x = self.pre_highway(x)
        for h in self.highways:
            x = h(x)

        # And then the RNN
        x, _ = self.rnn(x)
        return x


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # shape: [T, N]
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


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
                               kernel_size=conv1_kernel, stride=1, padding=conv1_kernel // 2)
        self.conv2 = nn.Conv1d(in_channels=d_fft, out_channels=d_model,
                               kernel_size=conv2_kernel, stride=1, padding=conv2_kernel // 2)
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
                src_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # shape: [N, T]
        x = x.transpose(0, 1)  # shape: [T, N]
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_pad_mask=src_pad_mask)
        x = self.norm(x)
        x = x.transpose(0, 1)
        return x


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
