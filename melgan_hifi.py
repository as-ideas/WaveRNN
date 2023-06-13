from typing import Tuple, Dict, Any

import torch
from torch.nn import Module, ModuleList, Sequential, LeakyReLU, Tanh, Conv1d

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm

MAX_WAV_VALUE = 32768.0
from scipy.signal import get_window


window = torch.from_numpy(get_window('hann', 16, fftbins=True).astype(np.float32))


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ResBlock(nn.Module):

    def __init__(self, channel, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(get_padding(kernel_size=kernel_size, dilation=dilations[i])),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=kernel_size, dilation=dilations[i])),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
            for i in range(len(dilations))
        ])

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x

    def remove_weight_norm(self):
        for block in self.blocks:
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])


class ResStack(nn.Module):

    def __init__(self, channel, kernel_sizes=(3, 7, 11), dilations=(1, 3, 5)):
        super(ResStack, self).__init__()

        self.blocks = nn.ModuleList([
            ResBlock(channel, dilations=dilations, kernel_size=kernel_sizes[i])
            for i in range(len(kernel_sizes))
        ])

    def forward(self, x):
        xs = None
        for block in self.blocks:
            if xs is None:
                xs = block(x)
            else:
                xs += block(x)
        return xs / len(self.blocks)

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)


class Generator(nn.Module):
    def __init__(self, mel_channel):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel

        self.generator = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mel_channel, 256, kernel_size=7, stride=1)),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4)),

            ResStack(128, kernel_sizes=(3, 7, 11)),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=16, stride=8, padding=4)),

            ResStack(64, kernel_sizes=(3, 7, 11)),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)),

            ResStack(32, kernel_sizes=(3, 7, 11)),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)),

            ResStack(16, kernel_sizes=(3, 7, 11)),

            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(16, 1, kernel_size=7, stride=1)),
            nn.Tanh(),
        )


    def forward(self, mel):
        mel = (mel + 5.0) / 5.0 # roughly normalize spectrogram
        return self.generator(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self,
                  mel: torch.Tensor,
                  pad_steps: int = 10) -> tuple:
        with torch.no_grad():
            pad = torch.full((1, 80, pad_steps), -11.5129).to(mel.device)
            mel = torch.cat((mel, pad), dim=2)
        return self.forward(mel)


