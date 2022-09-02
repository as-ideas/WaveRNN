

import itertools
from dataclasses import dataclass
from torch.multiprocessing import Pool
from pathlib import Path

import torch
from dp.utils.io import pickle_binary
from tqdm import tqdm

from trainer.common import np_now
from utils.dataset import get_tts_datasets
from utils.display import *
from utils.dsp import DSP
from utils.duration_extractor import DurationExtractor
from utils.files import read_config, unpickle_binary
from utils.metrics import attention_score
from utils.paths import Paths
from utils.text.tokenizer import Tokenizer


@dataclass
class ProcessorResult:
    item_id: str
    align_score: float
    att_score: float


class Processor:

    def __init__(self,
                 tokenizer: Tokenizer,
                 text_dict: dict,
                 duration_extractor: DurationExtractor,
                 paths: Paths) -> None:
        self.duration_extractor = duration_extractor
        self.tokenizer = tokenizer
        self.text_dict = text_dict
        self.paths = paths

    def __call__(self, item: tuple) -> ProcessorResult:

        item_id, mel_len = item

        x = self.text_dict[item_id]
        x = self.tokenizer(x)
        mel = np.load(self.paths.mel / f'{item_id}.npy')
        mel = torch.from_numpy(mel).unsqueeze(0)
        x = torch.from_numpy(x).unsqueeze(0)
        att_npy = np.load(str(self.paths.att_pred / f'{item_id}.npy'), mmap_mode='r', allow_pickle=False)
        att = torch.from_numpy(att_npy)
        align_score, _ = attention_score(att.unsqueeze(0), mel_len, r=1)
        durs, att_score = self.duration_extractor(x=x, mel=mel, att=att)
        durs_npy = np_now(durs).astype(np.int)
        np.save(str(self.paths.data / f'alg_extr/{item_id}.npy'), durs_npy, allow_pickle=False)

        return ProcessorResult(
            item_id=item_id,
            align_score=align_score,
            att_score=att_score
        )


if __name__ == '__main__':
    config = read_config('config.yaml')
    dsp = DSP.from_config(config)
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])

    duration_extractor = DurationExtractor(
        silence_threshold=config['preprocessing']['silence_threshold'],
        silence_prob_shift=config['preprocessing']['silence_prob_shift'])

    print('Performing duration extraction...')
    att_score_dict = {}
    processor = Processor(duration_extractor=duration_extractor,
                          tokenizer=Tokenizer(),
                          text_dict=unpickle_binary(paths.data / 'text_dict.pkl'),
                          paths=paths)
    pool = Pool(processes=12)

    train_set = unpickle_binary(paths.data / 'train_dataset.pkl')
    val_set = unpickle_binary(paths.data / 'val_dataset.pkl')
    dataset = train_set + val_set
    pbar = tqdm(pool.imap_unordered(processor, dataset), total=len(val_set)+len(train_set))
    att_scores = []
    for res in pbar:
        att_score_dict[res.item_id] = (res.align_score, res.att_score)
        att_scores.append(res.att_score)
        pbar.set_description(f'Avg align score: {sum(att_scores) / len(att_scores)}', refresh=True)

    pickle_binary(att_score_dict, paths.data / 'att_score_dict.pkl')
    print('done.')



