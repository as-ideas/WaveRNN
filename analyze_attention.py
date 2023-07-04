import math

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import torch

from models.tacotron import Tacotron
from utils.dataset import get_taco_dataloaders
from utils.display import plot_attention
from utils.files import read_config
from utils.metrics import attention_score
from utils.paths import Paths
import argparse
import itertools
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from duration_extraction.duration_extraction_pipe import DurationExtractionPipeline
from duration_extraction.duration_extractor import DurationExtractor
from models.tacotron import Tacotron
from trainer.common import to_device
from trainer.taco_trainer import TacoTrainer
from utils.checkpoints import restore_checkpoint
from utils.dataset import get_taco_dataloaders
from utils.display import *
from utils.dsp import DSP
from utils.files import pickle_binary, unpickle_binary, read_config
from utils.paths import Paths



if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = read_config('configs/multispeaker.yaml')

    paths = Paths(config['data_path'], config['tts_model_id'])
    print('\nInitialising Tacotron Model...\n')
    model = Tacotron.from_config(config).to(device)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint(model=model, optim=optimizer,
                       path=paths.taco_checkpoints / 'latest_model.pt',
                       device=device)

    train_cfg = config['tacotron']['training']

    train_set, val_set = get_taco_dataloaders(paths, 1, model.r, **train_cfg['filter'])

    model.eval()

    duration_extractor = DurationExtractor(silence_threshold=-11, silence_prob_shift=0)

    for batch in tqdm(train_set, total=len(train_set)):

        with torch.no_grad():
            out = model(batch)

        att = out['att']
        att_aligner = out['att_aligner'].softmax(-1)
        item_id = str(batch['item_id'][0])

        plot_attention(att.cpu().numpy())
        plt.savefig(f'/tmp/att/{item_id}.png')
        plt.clf()

        plot_attention(att_aligner.cpu().numpy())
        plt.savefig(f'/tmp/att/{item_id}_aligner.png')
        plt.clf()
        plt.close()

        T = att.size(1)
        N = att.size(2)
        dia_mat = torch.zeros((1, T, N))
        g = 0.2
        for t in range(T):
            for n in range(N):
                dia_mat[:, t, n] = math.exp(-(n / N - t / T) ** 2 / (2 * g ** 2))
        print()

        plot_attention(dia_mat.numpy())
        plt.savefig(f'/tmp/att/{item_id}_dia.png')
        plt.clf()
        plt.close()

        plot_attention((dia_mat* att_aligner).softmax(-1).numpy())
        plt.savefig(f'/tmp/att/{item_id}_dia_aligner.png')
        plt.clf()
        plt.close()
