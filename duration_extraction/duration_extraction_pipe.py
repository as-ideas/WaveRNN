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
from utils.dataset import BinnedLengthSampler, get_binned_taco_dataloader, DurationStats
from utils.files import unpickle_binary
from utils.metrics import attention_score
from utils.paths import Paths
from utils.text.tokenizer import Tokenizer


@dataclass
class DurationResult:
    item_id: str
    att_score: float
    align_score: float
    durations: np.array


class DurationCollator:

    def __call__(self, x: List[DurationResult]) -> DurationResult:
        if len(x) > 1:
            raise ValueError(f'Batch size must be 1! Found batch size: {len(x)}')
        return x[0]


class DurationExtractionDataset(Dataset):

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
        attention_npy = np.load(str(self.paths.att_pred / f'{item_id}.npy'))
        attention = torch.from_numpy(attention_npy)
        mel_len = mel.shape[-1]
        mel_len = torch.tensor(mel_len).unsqueeze(0)
        align_score, _ = attention_score(attention.unsqueeze(0), mel_len, r=1)
        align_score = float(align_score)
        durations, att_score = self.duration_extractor(x=x, mel=mel, attention=attention)
        att_score = float(att_score)
        durations_npy = durations.cpu().numpy()
        if np.sum(durations_npy) != mel_len:
            print(f'WARNINNG: Sum of durations did not match mel length for item {item_id}!')
        return DurationResult(item_id=item_id, att_score=att_score,
                              align_score=align_score, durations=durations_npy)

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
                           max_batch_size: int = 1) -> float:
        """
        Performs tacotron inference and stores the attention matrices as npy arrays in paths.data.att_pred.
        Returns average attention score.

        Args:
            model: Tacotron model to use for attention extraction.
            batch_size: Batch size to use for tacotron inference.

        Returns: Mean attention score. The attention matrices are saved as numpy arrays in paths.att_pred.

        """

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        dataloader = get_binned_taco_dataloader(paths=self.paths, max_batch_size=max_batch_size)

        sum_items = 0
        sum_att_score = 0
        pbar = tqdm(dataloader, total=len(dataloader), smoothing=0.01)
        for i, batch in enumerate(pbar, 1):
            batch = to_device(batch, device=device)
            with torch.no_grad():
                out = model(batch)
            attention_batch = out['att']
            _, att_score = attention_score(attention_batch, batch['mel_len'], r=1)
            sum_att_score += att_score.sum()
            B = batch['x_len'].size(0)
            sum_items += B
            for b in range(B):
                x_len = batch['x_len'][b].cpu()
                mel_len = batch['mel_len'][b].cpu()
                item_id = batch['item_id'][b]
                attention = attention_batch[b, :mel_len, :x_len].cpu()
                np.save(self.paths.att_pred / f'{item_id}.npy', attention.numpy(), allow_pickle=False)
            pbar.set_description(f'Avg attention score: {sum_att_score / sum_items}', refresh=True)

        return sum_att_score / len(dataloader)

    def extract_durations(self,
                          num_workers: int = 0,
                          sampler_bin_size: int = 1) -> Dict[str, DurationStats]:
        """
        Extracts durations from saved attention matrices.

        Args:
            num_workers: Number of workers for multiprocessing.
            sampler_bin_size: Bin size of BinnedLengthSampler.
            Should be greater than one (but much less than length of dataset) for optimal performance.

        Returns: Dictionary containing the attention scores for each item id.
        The durations are saved as numpy arrays in paths.alg.
        """

        train_set = unpickle_binary(self.paths.train_dataset)
        val_set = unpickle_binary(self.paths.val_dataset)
        text_dict = unpickle_binary(self.paths.text_dict)
        dataset = train_set + val_set
        dataset = [(file_id, mel_len) for file_id, mel_len in dataset
                   if (self.paths.att_pred / f'{file_id}.npy').is_file()]
        len_orig = len(dataset)
        data_ids, mel_lens = list(zip(*dataset))
        self.logger.info(f'Found {len(data_ids)} / {len_orig} '
                         f'alignment files in {self.paths.att_pred}')

        duration_stats = {}
        sum_att_score = 0

        dataset = DurationExtractionDataset(
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

        pbar = tqdm(dataset, total=len(dataset), smoothing=0.01)

        for i, res in enumerate(pbar, 1):
            sum_att_score += res.att_score
            pbar.set_description(f'Avg duration attention score: {sum_att_score / i}', refresh=True)
            max_consecutive_ones = self._get_max_consecutive_ones(res.durations)
            max_duration = np.max(res.durations)
            duration_stats[res.item_id] = DurationStats(att_align_score=res.align_score,
                                                        att_sharpness_score=res.att_score,
                                                        max_consecutive_ones=max_consecutive_ones,
                                                        max_duration=max_duration)
            np.save(self.paths.alg / f'{res.item_id}.npy', res.durations.astype(int), allow_pickle=False)

        return duration_stats

    @staticmethod
    def _get_max_consecutive_ones(durations: np.array) -> int:
        max_count = 0
        count = 0
        for d in durations:
            if d == 1:
                count += 1
            else:
                max_count = max(max_count, count)
                count = 0
        return max(max_count, count)
