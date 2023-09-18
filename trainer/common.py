from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader


class TTSSession:

    def __init__(self,
                 index: int,
                 r: int,
                 lr: int,
                 max_step: int,
                 bs: int,
                 train_set: DataLoader,
                 val_set: DataLoader) -> None:
        """ Container for TTS training variables. """

        self.index = index
        self.r = r
        self.lr = lr
        self.max_step = max_step
        self.bs = bs
        self.train_set = train_set
        self.val_set = val_set
        self.val_sample = next(iter(val_set))


class VocSession:

    def __init__(self,
                 index: int,
                 lr: int,
                 max_step: int,
                 bs: int,
                 train_set: DataLoader,
                 val_set: list,
                 val_set_samples: list) -> None:
        """ Container for WaveRNN training variables. """

        self.index = index
        self.lr = lr
        self.max_step = max_step
        self.bs = bs
        self.train_set = train_set
        self.val_set = val_set
        self.val_set_samples = val_set_samples


class Averager:

    def __init__(self) -> None:
        self.count = 0
        self.val = 0.

    def add(self, val: float) -> None:
        self.val += float(val)
        self.count += 1

    def reset(self) -> None:
        self.val = 0.
        self.count = 0

    def get(self) -> float:
        return self.val / self.count if self.count > 0. else 0.


class MaskedL1(torch.nn.Module):

    def forward(self, x, target, lens):
        target.requires_grad = False
        max_len = target.size(2)
        mask = pad_mask(lens, max_len)
        mask = mask.unsqueeze(1).expand_as(x)
        loss = F.l1_loss(
            x * mask, target * mask, reduction='sum')
        return loss / mask.sum()


class ForwardSumLoss(torch.nn.Module):

    def __init__(self, blank_logprob=-1):
        super(ForwardSumLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = torch.nn.CTCLoss(zero_infinity=True)

    def forward(self,
                attn_logprob: torch.Tensor,
                text_lens: torch.Tensor,
                mel_lens: torch.Tensor) -> torch.Tensor:

        # The CTC loss module assumes the existence of a blank token
        # that can be optionally inserted anywhere in the sequence for
        # a fixed probability.
        # A row must be added to the attention matrix to account for this
        attn_logprob_padded = F.pad(input=attn_logprob,
                                    pad=(1, 0, 0, 0, 0, 0),
                                    value=self.blank_logprob)
        batch_size = attn_logprob.size(0)
        steps = attn_logprob.size(-1)
        target_seq = torch.arange(1, steps+1).expand(batch_size, steps)
        attn_logprob_padded = attn_logprob_padded.permute(1, 0, 2)
        attn_logprob_padded = attn_logprob_padded.log_softmax(-1)
        cost = self.CTCLoss(attn_logprob_padded,
                            target_seq,
                            input_lengths=mel_lens,
                            target_lengths=text_lens)
        return cost

# Adapted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def pad_mask(lens, max_len):
    batch_size = lens.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range = seq_range.unsqueeze(0)
    seq_range = seq_range.expand(batch_size, max_len)
    if lens.is_cuda:
        seq_range = seq_range.cuda()
    lens = lens.unsqueeze(1)
    lens = lens.expand_as(seq_range)
    mask = seq_range < lens
    return mask.float()


def new_guided_attention_matrix(attention: torch.Tensor, g: float) -> torch.Tensor:
    T = attention.size(1)
    N = attention.size(2)
    t_vals = torch.arange(T, device=attention.device, dtype=attention.dtype)
    n_vals = torch.arange(N, device=attention.device, dtype=attention.dtype)
    t_diff = t_vals[:, None] / T - n_vals[None, :] / N
    dia_mat = torch.exp(-t_diff**2 / (2 * g**2)).unsqueeze(0)
    return dia_mat


def to_device(batch: Dict[str, torch.tensor],
              device: torch.device) -> Dict[str, torch.tensor]:
    output = {}
    for key, val in batch.items():
        val = val.to(device) if torch.is_tensor(val) else val
        output[key] = val
    return output


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()
