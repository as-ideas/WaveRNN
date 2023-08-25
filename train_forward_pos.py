import argparse
import itertools
import os
import subprocess
from pathlib import Path
from random import Random
from typing import Union, List, Dict
import numpy as np
import torch
import pandas as pd
import tqdm
from dp.training.dataset import PhonemizerDataset, BinnedLengthSampler
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from models.common_layers import make_token_len_mask
from models.fast_pitch import FastPitch
from models.forward_tacotron import ForwardTacotron
from models.multi_forward_tacotron import SeriesPredictor, MultiForwardTacotron
from trainer.common import to_device
from utils.checkpoints import restore_checkpoint, init_tts_model, save_checkpoint
from utils.display import *
from utils.dsp import DSP
from utils.files import read_config
from utils.paths import Paths
from utils.text.tokenizer import Tokenizer


# From https://github.com/fatchord/WaveRNN/blob/master/utils/dataset.py
class BinnedLengthSampler(DistributedSampler):

    def __init__(self, phoneme_lens: List[int], batch_size: int, bin_size: int, seed=42) -> None:
        _, self.idx = torch.sort(torch.tensor(phoneme_lens))
        self.batch_size = batch_size
        self.bin_size = bin_size
        self.random = Random(seed)
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        idx = self.idx.numpy()
        bins = []
        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            self.random.shuffle(this_bin)
            bins += [this_bin]
        self.random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)
        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            self.random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])
        return iter(torch.Tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)

class PosCollator:

    def __call__(self, batch) -> Dict[str, torch.tensor]:
        x = pad_sequence([b['x'] for b in batch], batch_first=True)
        pos = pad_sequence([b['pos'] for b in batch], batch_first=True)
        x_len = [b['x_len'] for b in batch]
        return {'x': x, 'pos': pos, 'x_len': x_len}


class PosDataset(Dataset):

    def __init__(self,
                 text_pos: List[tuple],
                 pos_dict: dict,
                 tokenizer: Tokenizer):
        self.text_pos = text_pos
        self.tokenizer = tokenizer
        self.pos_dict = pos_dict

    def __getitem__(self, index: int) -> dict:
        text, pos = self.text_pos[index]
        x = self.tokenizer(text)
        pos = [self.pos_dict[s] for s in pos]

        assert len(x) == len(pos), f'noneq len: {index}'

        x = torch.tensor(x).long()
        pos = torch.tensor(pos).long()

        return {'x': x, 'pos': pos, 'x_len': len(x)}

    def __len__(self):
        return len(self.text_pos)



if __name__ == '__main__':
    df_pos = pd.read_csv('/Users/cschaefe/datasets/nlp/pos/alpha_pos.tsv', sep='\t', encoding='utf-8')
    df_dep = pd.read_csv('/Users/cschaefe/datasets/nlp/pos/alpha_dep.tsv', sep='\t', encoding='utf-8')
    df_pos.dropna(inplace=True)
    df_dep.dropna(inplace=True)
    phon_pos = list(zip(df_pos['text_phonemized'], df_pos['text_phonemized_pos']))
    phon_dep = list(zip(df_dep['text_phonemized'], df_dep['text_phonemized_pos']))

    config = read_config('configs/multispeaker.yaml')
    paths = Paths(config['data_path'], config['tts_model_id'])
    model = MultiForwardTacotron.from_config(config)
    save_path = paths.forward_checkpoints / 'latest_model.pt'

    pos_dict = {}
    for _, pos in phon_pos:
        for p in pos:
            p = str(p)
            if p not in pos_dict:
                pos_dict[p] = len(pos_dict) + 1
    print(pos_dict)

    dep_dict = {}
    for _, pos in phon_dep:
        for p in pos:
            p = str(p)
            if p not in dep_dict:
                dep_dict[p] = len(dep_dict) + 1
    print(dep_dict)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)
    model = model.to(device)
    n_val = 32
    val_steps = 10
    batch_size = 32

    Random(42).shuffle(phon_pos)
    val_data_pos = phon_pos[:n_val]
    train_data_pos = phon_pos[n_val:]
    val_data_dep = phon_dep[:n_val]
    train_data_dep = phon_dep[n_val:]

    train_dataset_pos = PosDataset(train_data_pos, pos_dict, Tokenizer())
    val_dataset_pos = PosDataset(val_data_pos, pos_dict, Tokenizer())
    train_dataset_dep = PosDataset(train_data_dep, dep_dict, Tokenizer())
    val_dataset_dep = PosDataset(val_data_dep, dep_dict, Tokenizer())
    lens_pos = [len(p) for p, _ in train_data_pos]
    lens_dep = [len(p) for p, _ in train_data_dep]


    sampler_pos = BinnedLengthSampler(phoneme_lens=lens_pos, batch_size=batch_size, bin_size=3*batch_size)
    dataloader_pos = DataLoader(train_dataset_pos,
                            batch_size=batch_size,
                            collate_fn=PosCollator(),
                            sampler=sampler_pos,
                            drop_last=True)

    sampler_dep = BinnedLengthSampler(phoneme_lens=lens_dep, batch_size=batch_size, bin_size=3*batch_size)
    dataloader_dep = DataLoader(train_dataset_dep,
                            batch_size=batch_size,
                            collate_fn=PosCollator(),
                            sampler=sampler_dep,
                            drop_last=True)


    optim = torch.optim.Adam(model.pos_pred.parameters(), lr=1e-3)
    forward_optim = torch.optim.Adam(model.parameters())
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    step = 0

    sw = SummaryWriter('checkpoints/pos_tagger')

    for epoch in range(1000):
        for batch_pos, batch_dep in tqdm.tqdm(zip(dataloader_pos, dataloader_dep), total=len(dataloader_pos)):
            model.step += 1
            batch_pos = to_device(batch_pos, device)
            batch_dep = to_device(batch_dep, device)
            out_pos, _ = model.pos_pred(batch_pos['x'])
            _, out_dep = model.pos_pred(batch_dep['x'])

            loss_pos = ce_loss(out_pos.transpose(1, 2), batch_pos['pos'])
            loss_dep = ce_loss(out_dep.transpose(1, 2), batch_dep['pos'])

            loss = 0.5 * loss_pos + 0.5 * loss_dep

            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1

            sw.add_scalar('loss', loss, global_step=step)

            if step % 10 == 0:
                example_pred = torch.argmax(out_pos[0], dim=-1)
                example_target = batch_pos['pos'][0]

                print(example_pred)
                print(example_target)

            if step % val_steps == 0:
                model.eval()
                val_acc = 0
                for batch_pos, batch_dep in tqdm.tqdm(zip(val_dataset_pos, val_dataset_dep), total=len(val_dataset_pos)):
                    batch_pos = to_device(batch_pos, device)
                    x = batch_pos['x'].unsqueeze(0)
                    pos = batch_pos['pos']
                    out_pos, _ = model.pos_pred(x)
                    example_pred = torch.argmax(out_pos[0], dim=-1)
                    matching = example_pred == pos
                    tp = sum(matching)
                    acc = tp / pos.size(-1)
                    val_acc += acc
                    batch_dep = to_device(batch_dep, device)
                    x = batch_dep['x'].unsqueeze(0)
                    pos = batch_dep['pos']
                    _, out_dep = model.pos_pred(x)
                    example_pred = torch.argmax(out_dep[0], dim=-1)
                    matching = example_pred == pos
                    tp = sum(matching)
                    acc = tp / pos.size(-1)
                    val_acc += acc

                val_acc /= 2. * len(val_dataset_pos)
                sw.add_scalar('val_acc', val_acc, global_step=step)
                print('VAL ACC: ', step, val_acc)

                print('checkpointing to ', save_path)
                save_checkpoint(model, forward_optim, config, save_path)
                model.train()