import random
from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import List, Tuple, Iterator

import torch
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from utils.dsp import *
from utils.files import unpickle_binary
from utils.paths import Paths
from utils.text.tokenizer import Tokenizer

SHUFFLE_SEED = 42


@dataclass
class DurationStats:
    att_sharpness_score: float
    att_align_score: float
    max_consecutive_ones: int
    max_duration: int


class DataFilter:

    def __init__(self,
                 duration_stats: Dict[str, DurationStats],
                 min_attention_alignment: float,
                 min_attention_sharpness: float,
                 max_consecutive_duration_ones: int,
                 max_duration: int):
        self._duration_stats = duration_stats
        self._att_min_alignment = min_attention_alignment
        self._att_min_sharpness = min_attention_sharpness
        self._max_consecutive_duration_ones = max_consecutive_duration_ones
        self._max_duration = max_duration

    def __call__(self, dataset: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        dataset_filtered = []
        for item_id, mel_len in dataset:
            dur_stat: DurationStats = self._duration_stats[item_id]
            if all([dur_stat.att_align_score >= self._att_min_alignment,
                    dur_stat.att_sharpness_score >= self._att_min_sharpness,
                    dur_stat.max_consecutive_ones <= self._max_consecutive_duration_ones,
                    dur_stat.max_duration <= self._max_duration]):
                dataset_filtered.append((item_id, mel_len))
        return dataset_filtered


class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        # Need to change to numpy since there's a bug in random.shuffle(tensor)
        # TODO: Post an issue on pytorch repo
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)


class TacoDataset(Dataset):

    def __init__(self,
                 paths: Paths,
                 dataset_ids: List[str],
                 text_dict: Dict[str, str],
                 speaker_dict: Dict[str, str],
                 tokenizer: Tokenizer) -> None:
        self.paths = paths
        self.metadata = dataset_ids
        self.text_dict = text_dict
        self.speaker_dict = speaker_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> Dict[str, torch.tensor]:
        item_id = self.metadata[index]
        text = self.text_dict[item_id]
        speaker_name = self.speaker_dict[item_id]
        x = self.tokenizer(text)
        mel = np.load(str(self.paths.mel/f'{item_id}.npy'))
        mel_len = mel.shape[-1]
        speaker_emb = np.load(str(self.paths.speaker_emb/f'{item_id}.npy'))
        return {'x': x, 'mel': mel, 'item_id': item_id,
                'mel_len': mel_len, 'x_len': len(x),
                'speaker_emb': speaker_emb, 'speaker_name': speaker_name}

    def __len__(self):
        return len(self.metadata)


class ForwardDataset(Dataset):

    def __init__(self,
                 paths: Paths,
                 dataset_ids: List[str],
                 text_dict: Dict[str, str],
                 speaker_dict: Dict[str, str],
                 tokenizer: Tokenizer):
        self.paths = paths
        self.metadata = dataset_ids
        self.text_dict = text_dict
        self.speaker_dict = speaker_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> Dict[str, torch.tensor]:
        item_id = self.metadata[index]
        text = self.text_dict[item_id]
        speaker_name = self.speaker_dict[item_id]
        x = self.tokenizer(text)
        mel = np.load(str(self.paths.mel/f'{item_id}.npy'))
        mel_len = mel.shape[-1]
        dur = np.load(str(self.paths.alg/f'{item_id}.npy'))
        pitch = np.load(str(self.paths.phon_pitch/f'{item_id}.npy'))
        energy = np.load(str(self.paths.phon_energy/f'{item_id}.npy'))
        speaker_emb = np.load(str(self.paths.speaker_emb/f'{item_id}.npy'))
        pitch_cond = np.ones(pitch.shape)
        pitch_cond[pitch != 0] = 2

        return {'x': x, 'mel': mel, 'item_id': item_id, 'x_len': len(x),
                'mel_len': mel_len, 'dur': dur, 'pitch': pitch, 'energy': energy,
                'speaker_emb': speaker_emb, 'pitch_cond': pitch_cond, 'speaker_name': speaker_name}

    def __len__(self):
        return len(self.metadata)


class BinnedTacoDataLoader:
    """
    Special dataloader to allow tacotron inference on batches with equal input length per batch.
    This is used to safely generate attention matrices for extracting phoneme durations as there
    is no input padding needed.
    """

    def __init__(self,
                 paths: Paths,
                 dataset: List[Tuple[str, int]],
                 max_batch_size: int = 8) -> None:
        """
        Initializes the dataloader.

        Args:
            pahts: Paths object containing data pahts.
            dataset: List of Tuples with (file_id, mel_length),
            max_batch_size: Maximum allowed batch size.
        """

        tokenizer = Tokenizer()
        file_id_text_lens = []
        text_dict = unpickle_binary(paths.text_dict)
        speaker_dict = unpickle_binary(paths.speaker_dict)
        for item_id, _ in dataset:
            toks = tokenizer(text_dict[item_id])
            file_id_text_lens.append((item_id, len(toks)))

        file_id_text_lens.sort(key=lambda x: x[1])
        dataset_ids = [file_id for file_id, _ in file_id_text_lens]
        dataset_lens = [text_len for _, text_len in file_id_text_lens]
        dataset_lens = np.array(dataset_lens, dtype=int)
        consecutive_split_points = np.where(np.diff(dataset_lens, append=0, prepend=0) != 0)[0]
        dataset_indices = list(range(len(dataset)))
        all_batches = []

        for a, b in zip(consecutive_split_points[:-1], consecutive_split_points[1:]):
            big_batch = dataset_indices[a:b]
            batches = list(_batchify(big_batch, batch_size=max_batch_size))
            all_batches.extend(batches)

        Random(SHUFFLE_SEED).shuffle(all_batches)
        self.all_batches = all_batches
        self.taco_dataset = TacoDataset(paths=paths, dataset_ids=dataset_ids,
                                        text_dict=text_dict, speaker_dict=speaker_dict,
                                        tokenizer=tokenizer)
        self.collator = TacoCollator(r=1)

    def __iter__(self) -> Iterator:
        for batch in self.all_batches:
            batch = [self.taco_dataset[i] for i in batch]
            batch = self.collator(batch)
            yield batch

    def __len__(self) -> int:
        return len(self.all_batches)


class TacoCollator:

    def __init__(self, r: int) -> None:
        self.r = r

    def __call__(self, batch: List[Dict[str, Union[str, torch.tensor]]]) -> Dict[str, torch.tensor]:
        x_len = [b['x_len'] for b in batch]
        x_len = torch.tensor(x_len)
        max_x_len = max(x_len)
        text = [_pad1d(b['x'], max_x_len) for b in batch]
        text = _stack_to_tensor(text).long()
        spec_lens = [b['mel_len'] for b in batch]
        max_spec_len = max(spec_lens) + 1
        if max_spec_len % self.r != 0:
            max_spec_len += self.r - max_spec_len % self.r
        mel = [_pad2d(b['mel'], max_spec_len) for b in batch]
        mel = _stack_to_tensor(mel)
        item_id = [b['item_id'] for b in batch]
        speaker_name = [b['speaker_name'] for b in batch]
        mel_lens = [b['mel_len'] for b in batch]
        mel_lens = torch.tensor(mel_lens)
        speaker_emb = [b['speaker_emb'] for b in batch]
        speaker_emb = _stack_to_tensor(speaker_emb)

        return {'x': text, 'mel': mel, 'item_id': item_id,
                'x_len': x_len, 'mel_len': mel_lens,
                'speaker_emb': speaker_emb, 'speaker_name': speaker_name}


class ForwardCollator:

    def __init__(self, taco_collator: TacoCollator) -> None:
        self.taco_collator = taco_collator

    def __call__(self, batch: List[Dict[str, Union[str, torch.tensor]]]) -> Dict[str, torch.tensor]:
        output = self.taco_collator(batch)
        x_len = [b['x_len'] for b in batch]
        x_len = torch.tensor(x_len)
        max_x_len = max(x_len)
        dur = [_pad1d(b['dur'][:max_x_len], max_x_len) for b in batch]
        dur = _stack_to_tensor(dur).float()
        pitch = [_pad1d(b['pitch'][:max_x_len], max_x_len) for b in batch]
        pitch = _stack_to_tensor(pitch).float()
        energy = [_pad1d(b['energy'][:max_x_len], max_x_len) for b in batch]
        energy = _stack_to_tensor(energy).float()
        pitch_cond = [_pad1d(b['pitch_cond'][:max_x_len], max_x_len) for b in batch]
        pitch_cond = _stack_to_tensor(pitch_cond).long()
        output.update({
            'pitch': pitch,
            'energy': energy,
            'dur': dur,
            'pitch_cond': pitch_cond
        })
        return output


def get_taco_dataloaders(paths: Paths,
                         batch_size: int,
                         r: int,
                         max_mel_len: int,
                         filter_duration_stats: bool,
                         min_attention_alignment: float,
                         min_attention_sharpness: float,
                         max_consecutive_ones: int,
                         max_duration: int,
                         num_workers=0) -> Tuple[DataLoader, DataLoader]:

    """Returns training and validation dataloaders.

    Args:
        paths (Paths): An instance of the Paths class containing file paths.
        batch_size (int): The batch size for the dataloaders.
        max_mel_len (int, optional): The maximum length of the mel spectrograms.
        filter_duration_stats (bool): Whether to filter accordint to below duration stats.
        min_attention_alignment (float): The minimum attention alignment value.
        min_attention_sharpness (float): The minimum attention sharpness value.
        max_consecutive_ones (int): The maximum number of consecutive ones in the alignment.
        max_duration (int): The maximum duration of the audio in seconds.
        num_workers (int): The number of worker processes for loading data. Defaults to 0.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple of two PyTorch DataLoaders for training and validation.
    """

    train_data, val_data = _get_filtered_datasets(paths=paths,
                                                  filter_duration_stats=filter_duration_stats,
                                                  max_mel_len=max_mel_len,
                                                  min_attention_alignment=min_attention_alignment,
                                                  min_attention_sharpness=min_attention_sharpness,
                                                  max_consecutive_ones=max_consecutive_ones,
                                                  max_duration=max_duration)

    train_ids, train_lens = zip(*train_data)
    val_ids, val_lens = zip(*val_data)

    tokenizer = Tokenizer()
    text_dict = unpickle_binary(paths.text_dict)
    speaker_dict = unpickle_binary(paths.speaker_dict)

    train_dataset = TacoDataset(paths=paths, dataset_ids=train_ids,
                                text_dict=text_dict, speaker_dict=speaker_dict,
                                tokenizer=tokenizer)
    val_dataset = TacoDataset(paths=paths, dataset_ids=val_ids,
                              text_dict=text_dict, speaker_dict=speaker_dict,
                              tokenizer=tokenizer)
    train_sampler = BinnedLengthSampler(train_lens, batch_size, batch_size * 3)
    collator = TacoCollator(r=r)

    train_set = DataLoader(train_dataset,
                           collate_fn=collator,
                           batch_size=batch_size,
                           sampler=train_sampler,
                           num_workers=num_workers,
                           pin_memory=True)

    val_set = DataLoader(val_dataset,
                         collate_fn=collator,
                         batch_size=batch_size,
                         sampler=None,
                         num_workers=0,
                         shuffle=False,
                         pin_memory=True)

    return train_set, val_set


def get_forward_dataloaders(paths: Paths,
                            batch_size: int,
                            max_mel_len: int,
                            filter_duration_stats: bool,
                            min_attention_alignment: float,
                            min_attention_sharpness: float,
                            max_consecutive_ones: int,
                            max_duration: int,
                            num_workers=0) -> Tuple[DataLoader, DataLoader]:
    """Returns training and validation dataloaders.

    Args:
        paths (Paths): An instance of the Paths class containing file paths.
        batch_size (int): The batch size for the dataloaders.
        max_mel_len (int, optional): The maximum length of the mel spectrograms.
        filter_duration_stats (bool): Whether to filter accordint to below duration stats.
        min_attention_alignment (float): The minimum attention alignment value.
        min_attention_sharpness (float): The minimum attention sharpness value.
        max_consecutive_ones (int): The maximum number of consecutive ones in the alignment.
        max_duration (int): The maximum duration of the audio in seconds.
        num_workers (int): The number of worker processes for loading data. Defaults to 0.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple of two PyTorch DataLoaders for training and validation.
    """

    tokenizer = Tokenizer()

    train_data, val_data = _get_filtered_datasets(paths=paths,
                                                  filter_duration_stats=filter_duration_stats,
                                                  max_mel_len=max_mel_len,
                                                  min_attention_alignment=min_attention_alignment,
                                                  min_attention_sharpness=min_attention_sharpness,
                                                  max_consecutive_ones=max_consecutive_ones,
                                                  max_duration=max_duration)

    text_dict = unpickle_binary(paths.text_dict)
    speaker_dict = unpickle_binary(paths.speaker_dict)
    speaker_dict = {item_id: speaker for item_id, speaker in speaker_dict.items() if item_id in speaker_dict}

    train_ids, train_lens = zip(*train_data)
    val_ids, val_lens = zip(*val_data)

    train_dataset = ForwardDataset(paths=paths, dataset_ids=train_ids,
                                   text_dict=text_dict, speaker_dict=speaker_dict,
                                   tokenizer=tokenizer)

    val_dataset = ForwardDataset(paths=paths, dataset_ids=val_ids,
                                 text_dict=text_dict, speaker_dict=speaker_dict,
                                 tokenizer=tokenizer)

    train_sampler = BinnedLengthSampler(train_lens, batch_size, batch_size * 3)
    collator = ForwardCollator(taco_collator=TacoCollator(r=1))

    train_set = DataLoader(train_dataset,
                           collate_fn=collator,
                           batch_size=batch_size,
                           sampler=train_sampler,
                           num_workers=num_workers,
                           pin_memory=True)

    val_set = DataLoader(val_dataset,
                         collate_fn=collator,
                         batch_size=batch_size,
                         sampler=None,
                         num_workers=0,
                         shuffle=False,
                         pin_memory=True)

    return train_set, val_set


def get_binned_taco_dataloader(paths: Paths, max_batch_size: int = 8) -> BinnedTacoDataLoader:
    train_data = unpickle_binary(paths.train_dataset)
    val_data = unpickle_binary(paths.val_dataset)
    dataset = train_data + val_data
    return BinnedTacoDataLoader(paths=paths, dataset=dataset, max_batch_size=max_batch_size)


def _get_filtered_datasets(paths: Paths,
                           max_mel_len: int,
                           filter_duration_stats: bool,
                           min_attention_alignment: float,
                           min_attention_sharpness: float,
                           max_consecutive_ones: int,
                           max_duration: int) -> Tuple[List[tuple], List[tuple]]:

    train_data = unpickle_binary(paths.train_dataset)
    val_data = unpickle_binary(paths.val_dataset)
    speaker_dict = unpickle_binary(paths.speaker_dict)

    train_data = _filter_max_len(train_data, max_mel_len)
    val_data = _filter_max_len(val_data, max_mel_len)

    if filter_duration_stats:
        duration_stats = unpickle_binary(paths.duration_stats)
        data_filter = DataFilter(duration_stats=duration_stats,
                                 min_attention_alignment=min_attention_alignment,
                                 min_attention_sharpness=min_attention_sharpness,
                                 max_consecutive_duration_ones=max_consecutive_ones,
                                 max_duration=max_duration)

        speaker_counts_orig = Counter([speaker_dict[item_id] for item_id, _ in train_data + val_data if item_id in speaker_dict])
        train_data = data_filter(train_data)
        val_data = data_filter(val_data)
        speaker_counts_filtered = Counter([speaker_dict[item_id] for item_id, _ in train_data + val_data if item_id in speaker_dict])

        table = [(speaker,
                  count,
                  speaker_counts_orig[speaker],
                  speaker_counts_orig[speaker]-count)
                 for speaker, count in speaker_counts_filtered.most_common()]
        print(tabulate(table, headers=('speaker', 'files used', 'files total', 'files removed')))
        num_files = sum(speaker_counts_filtered.values())
        num_filtered = sum(speaker_counts_orig.values()) - num_files
        print(f'\nUsing {num_files} files, removed {num_filtered}')

    return train_data, val_data


def _filter_max_len(dataset: List[tuple], max_mel_len: int) -> List[tuple]:
    if max_mel_len is None:
        return dataset
    return [(id, len) for id, len in dataset if len <= max_mel_len]


def _stack_to_tensor(x: List[np.array]) -> torch.Tensor:
    x = np.stack(x)
    x = torch.tensor(x)
    return x


def _pad1d(x, max_len) -> np.array:
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def _pad2d(x, max_len) -> np.array:
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), constant_values=-11.5129, mode='constant')


def _batchify(input: List[Any], batch_size: int) -> List[List[Any]]:
    input_len = len(input)
    output = []
    for i in range(0, input_len, batch_size):
        batch = input[i:min(i + batch_size, input_len)]
        output.append(batch)
    return output
