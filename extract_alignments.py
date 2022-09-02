

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
from utils.files import read_config
from utils.metrics import attention_score
from utils.paths import Paths


@dataclass
class ProcessorResult:
    item_id: str
    align_score: float
    att_score: float


class Processor:

    def __init__(self,
                 duration_extractor: DurationExtractor,
                 att_pred_path: Path,
                 alg_path: Path) -> None:
        self.duration_extractor = duration_extractor
        self.att_pred_path = att_pred_path
        self.alg_path = alg_path

    def __call__(self, batch: dict) -> ProcessorResult:
        x = batch['x'][0]
        mel_len = batch['mel_len'][0]
        item_id = batch['item_id'][0]
        mel = batch['mel'][0, :, :mel_len]
        att_npy = np.load(str(self.att_pred_path / f'{item_id}.npy'))
        att = torch.from_numpy(att_npy)
        del att_npy

        align_score, _ = attention_score(att.unsqueeze(0), batch['mel_len'], r=1)
        durs, att_score = self.duration_extractor(x=x, mel=mel, att=att)
        durs_npy = np_now(durs).astype(np.int)
        np.save(str(self.alg_path / f'{item_id}.npy'), durs_npy, allow_pickle=False)

        del durs
        del durs_npy
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
                          att_pred_path=paths.att_pred,
                          alg_path=paths.alg)
    pool = Pool(processes=4)

    train_set, val_set = get_tts_datasets(paths.data, 1, 1,
                                          max_mel_len=None,
                                          filter_attention=False)
    dataset = itertools.chain(train_set, val_set)
    pbar = tqdm(pool.imap_unordered(processor, dataset), total=len(val_set)+len(train_set))
    att_scores = []
    for res in pbar:
        att_score_dict[res.item_id] = (res.align_score, res.att_score)
        att_scores.append(res.att_score)
        pbar.set_description(f'Avg align score: {sum(att_scores) / len(att_scores)}', refresh=True)

    pickle_binary(att_score_dict, paths.data / 'att_score_dict.pkl')
    print('done.')



