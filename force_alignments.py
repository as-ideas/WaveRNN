

import itertools
from dataclasses import dataclass
from pathlib import Path

import torch
from dp.utils.io import pickle_binary
from torch import optim
from torch.multiprocessing import Pool
from tqdm import tqdm

from models.tacotron import Tacotron
from trainer.common import np_now
from utils.checkpoints import restore_checkpoint
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
                 model: Tacotron,
                 duration_extractor: DurationExtractor,
                 alg_path: Path) -> None:
        self.model = model
        self.duration_extractor = duration_extractor
        self.alg_path = alg_path

    def __call__(self, batch: dict) -> ProcessorResult:
        with torch.no_grad():
            _, _, att_batch = self.model(batch['x'], batch['mel'], batch['speaker_emb'])
        x = batch['x'][0]
        mel_len = batch['mel_len'][0]
        item_id = batch['item_id'][0]
        mel = batch['mel'][0, :, :mel_len]
        att = att_batch[0, :mel_len, :]
        align_score, _ = attention_score(att_batch, batch['mel_len'], r=1)
        align_score = float(align_score[0])
        durs, att_score = self.duration_extractor(x=x, mel=mel, att=att)
        durs = np_now(durs).astype(np.int)
        np.save(str(self.alg_path / f'{item_id}.npy'), durs, allow_pickle=False)
        return ProcessorResult(
            item_id=item_id,
            align_score=align_score,
            att_score=att_score
        )



if __name__ == '__main__':

    config = read_config('config.yaml')
    dsp = DSP.from_config(config)
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])

    print('\nInitialising Tacotron Model...\n')
    device = torch.device('cpu')
    model = Tacotron.from_config(config).to(device)

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint(model=model, optim=optimizer,
                       path=paths.taco_checkpoints / 'latest_model.pt',
                       device=device)

    duration_extractor = DurationExtractor(
        silence_threshold=config['preprocessing']['silence_threshold'],
        silence_prob_shift=config['preprocessing']['silence_prob_shift'])

    processor = Processor(model=model, duration_extractor=duration_extractor,
                          alg_path=paths.alg)
    pool = Pool(processes=4)
    train_set, val_set = get_tts_datasets(paths.data, 1, model.r,
                                          max_mel_len=None,
                                          filter_attention=False)
    dataset = itertools.chain(train_set, val_set)
    att_score_dict = {}
    att_scores = []

    for res in tqdm(pool.imap_unordered(processor, val_set), total=len(train_set) + len(val_set)):
        att_score_dict[res.item_id] = (res.align_score, res.att_score)
        att_scores.append(res.att_score)

    pickle_binary(att_score_dict, paths.data / 'att_score_dict.pkl')



