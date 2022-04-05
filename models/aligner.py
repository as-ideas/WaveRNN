from pathlib import Path
from typing import Union, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, GRU

from models.common_layers import CBHG
from utils.text.symbols import phonemes

class Aligner(nn.Module):


    def __init__(self, num_chars):
        super().__init__()
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.embedding = Embedding(num_embeddings=num_chars, embedding_dim=512)

        self.text_encoder = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=512, out_channels=64, kernel_size=3, padding=1)
        )
        self.mel_encoder = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=512, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=512, out_channels=64, kernel_size=3, padding=1)
        )

        self.gru_1 = GRU(64, 64, bidirectional=True)
        self.gru_2 = GRU(64, 64, bidirectional=True)


    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.step += 1
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.text_encoder(x)

        m = self.mel_encoder(m)

        x = x.transpose(1, 2)
        m = m.transpose(1, 2)

        x, _ = self.gru_1(x)
        m, _ = self.gru_1(m)

        diff = x[:, None, :, :] - m[:, :, None, :]
        dist = -torch.linalg.norm(diff, ord=2, dim=-1)
        return dist


    def get_step(self) -> int:
        return self.step.data.item()
