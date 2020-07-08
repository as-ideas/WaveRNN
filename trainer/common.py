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
        self.index = index
        self.lr = lr
        self.max_step = max_step
        self.bs = bs
        self.train_set = train_set
        self.val_set = val_set
        self.val_set_samples = val_set_samples


class Averager:

    def __init__(self):
        self.count = 0
        self.val = 0.

    def add(self, val):
        self.val += float(val)
        self.count += 1

    def reset(self):
        self.val = 0.
        self.count = 0

    def get(self):
        return self.val / self.count


class MaskedL1(torch.nn.Module):

    def forward(self, x, target, lens):
        target.requires_grad = False
        max_len = target.size(1)
        mask = pad_mask(lens, max_len)
        mask = mask.unsqueeze(2).expand_as(x)
        loss = F.l1_loss(
            x * mask, target * mask, reduction='sum')
        return loss / mask.sum()


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
