import itertools
import logging
from dataclasses import dataclass
from logging import INFO
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from duration_extraction.duration_extractor import DurationExtractor
from models.tacotron import Tacotron
from trainer.common import to_device
from utils.dataset import get_tts_datasets, BinnedLengthSampler
from utils.files import unpickle_binary
from utils.metrics import attention_score
from utils.paths import Paths
from utils.text.tokenizer import Tokenizer


@dataclass
class DurationResult:
    item_id: str
    att_score: float
    align_score: float
    durs: np.array


class DurationCollator:

    def __call__(self, x: List[DurationResult]) -> DurationResult:
        if len(x) > 1:
            raise ValueError(f'Batch size must be 1! Found batch size: {len(x)}')
        return x[0]


class DurationDataset(Dataset):

    def __init__(self,
                 duration_extractor: DurationExtractor,
                 paths: Paths,
                 dataset_ids: List[str],
                 text_dict: Dict[str, str],
                 tokenizer: Tokenizer):
        self.metadata = dataset_ids
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.text_dict = text_dict
        self.duration_extractor = duration_extractor
        self.paths = paths

    def __getitem__(self, index: int) -> DurationResult:
        item_id = self.metadata[index]
        x = self.text_dict[item_id]
        x = self.tokenizer(x)
        mel = np.load(self.paths.mel / f'{item_id}.npy')
        mel = torch.from_numpy(mel)
        x = torch.tensor(x)
        att_npy = np.load(str(self.paths.att_pred / f'{item_id}.npy'))
        att = torch.from_numpy(att_npy)
        mel_len = mel.shape[-1]
        mel_len = torch.tensor(mel_len).unsqueeze(0)
        align_score, _ = attention_score(att.unsqueeze(0), mel_len, r=1)
        align_score = float(align_score)
        durs, att_score = self.duration_extractor(x=x, mel=mel, att=att)
        att_score = float(att_score)
        durs_npy = durs.cpu().numpy()
        if np.sum(durs_npy) != mel_len:
            print(f'WARNINNG: Sum of durations did not match mel length for item {item_id}!')
        return DurationResult(item_id=item_id, att_score=att_score,
                              align_score=align_score, durs=durs_npy)

    def __len__(self):
        return len(self.metadata)


class DurationExtractionPipeline:

    def __init__(self,
                 paths: Paths,
                 config: Dict[str, Any],
                 duration_extractor: DurationExtractor) -> None:
        self.paths = paths
        self.config = config
        self.duration_extractor = duration_extractor
        self.logger = logging.Logger(__name__, level=INFO)

    def extract_attentions(self,
                           model: Tacotron,
                           batch_size: int = 1) -> float:
        """
        Performs tacotron inference and stores the attention matrices as npy arrays in paths.data.att_pred.
        Returns average attention score.
        """

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        sum_att_score = 0
        train_set, val_set = get_tts_datasets(path=self.paths.data,
                                              batch_size=batch_size,
                                              r=1,
                                              max_mel_len=None,
                                              filter_attention=False,
                                              model_type='tacotron')

        dataset = itertools.chain(train_set, val_set)

        pbar = tqdm(dataset, total=len(val_set) + len(train_set))
        for i, batch in enumerate(pbar, 1):
            batch = to_device(batch, device=device)
            with torch.no_grad():
                _, _, att_batch = model(batch['x'], batch['mel'])
            _, att_score = attention_score(att_batch, batch['mel_len'], r=1)
            sum_att_score += att_score.sum()
            for b in range(batch['x_len'].size(0)):
                x_len = batch['x_len'][b].cpu()
                mel_len = batch['mel_len'][b].cpu()
                item_id = batch['item_id'][b]
                att = att_batch[b, :mel_len, :x_len].cpu()
                np.save(self.paths.att_pred / f'{item_id}.npy', att.numpy(), allow_pickle=False)
            pbar.set_description(f'Avg attention score: {sum_att_score / (i * batch_size)}', refresh=True)

        return sum_att_score / (len(train_set) + len(val_set))

    def extract_durations(self,
                          num_workers: int = 0,
                          sampler_bin_size: int = 1) -> Dict[str, Tuple[float, float]]:

        """
        Extracts durations from saved attention matrices, saves the durations as npy arrays
        and returns a dictionary with entries {file_id: (attention_alignment_score, attention_sharpness score)}
        """

        train_set = unpickle_binary(self.paths.data / 'train_dataset.pkl')
        val_set = unpickle_binary(self.paths.data / 'val_dataset.pkl')
        text_dict = unpickle_binary(self.paths.data / 'text_dict.pkl')
        dataset = train_set + val_set
        dataset = [(file_id, mel_len) for file_id, mel_len in dataset
                   if (self.paths.att_pred / f'{file_id}.npy').is_file()]
        len_orig = len(dataset)
        data_ids, mel_lens = list(zip(*dataset))
        self.logger.info(f'Found {len(data_ids)} / {len_orig} '
                         f'alignment files in {self.paths.att_pred}')
        att_score_dict = {}
        sum_att_score = 0

        dataset = DurationDataset(
            duration_extractor=self.duration_extractor,
            paths=self.paths, dataset_ids=data_ids,
            text_dict=text_dict, tokenizer=Tokenizer())

        dataset = DataLoader(dataset=dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=False,
                             collate_fn=DurationCollator(),
                             sampler=BinnedLengthSampler(lengths=mel_lens, batch_size=1, bin_size=sampler_bin_size),
                             num_workers=num_workers)

        pbar = tqdm(dataset, total=len(dataset))

        for i, res in enumerate(pbar, 1):
            pbar.set_description(f'Avg tuned attention score: {sum_att_score / i}', refresh=True)
            att_score_dict[res.item_id] = (res.align_score, res.att_score)
            sum_att_score += res.att_score
            np.save(self.paths.alg / f'{res.item_id}.npy', res.durs.astype(int), allow_pickle=False)

        return att_score_dict
