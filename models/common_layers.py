import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


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
                 dropout: float = 0.5,
                 residual_strength: float = 0.) -> None:
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
        self.res_lin = nn.Linear(channels, 2*channels)
        self.residual_strength = residual_strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        seq_len = x.size(-1)
        conv_bank = []

        # Convolution Bank
        for conv in self.conv1d_bank:
            c = conv(x) # Convolution
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
        x_res = self.res_lin(x)
        x, _ = self.rnn(x)
        x = x_res * self.residual_strength + x

        return x