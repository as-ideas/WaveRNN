import random
from random import Random
from typing import List, Tuple, Iterator

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from utils.dsp import *
from utils.files import unpickle_binary
from utils.text.tokenizer import Tokenizer


SHUFFLE_SEED = 42

###################################################################################
# WaveRNN/Vocoder Dataset #########################################################
###################################################################################


class VocoderDataset(Dataset):

    def __init__(self, path: Path, dataset_ids, train_gta=False) -> None:
        self.metadata = dataset_ids
        self.mel_path = path/'gta' if train_gta else path/'mel'
        self.quant_path = path/'quant'

    def __getitem__(self, index: int) -> Dict[str, np.array]:
        item_id = self.metadata[index]
        mel = np.load(str(self.mel_path/f'{item_id}.npy'))
        x = np.load(str(self.quant_path/f'{item_id}.npy'))
        return {'mel': mel, 'x': x}

    def __len__(self):
        return len(self.metadata)


def get_vocoder_datasets(path: Path,
                         batch_size: int,
                         train_gta: bool,
                         max_mel_len: int,
                         hop_length: int,
                         voc_pad: int,
                         voc_seq_len: int,
                         voc_mode: str,
                         bits: int,
                         num_gen_samples: int):
    train_data = unpickle_binary(path/'train_dataset.pkl')
    val_data = unpickle_binary(path/'val_dataset.pkl')
    train_ids, train_lens = zip(*filter_max_len(train_data, max_mel_len))
    val_ids, val_lens = zip(*filter_max_len(val_data, max_mel_len))
    train_dataset = VocoderDataset(path, train_ids, train_gta)
    val_dataset = VocoderDataset(path, val_ids, train_gta)
    voc_collator = VocCollator(hop_length=hop_length,
                               voc_pad=voc_pad,
                               voc_seq_len=voc_seq_len,
                               voc_mode=voc_mode,
                               bits=bits)
    train_set = DataLoader(train_dataset,
                           collate_fn=voc_collator,
                           batch_size=batch_size,
                           num_workers=0,
                           shuffle=True,
                           pin_memory=True)

    val_set = DataLoader(val_dataset,
                         collate_fn=voc_collator,
                         batch_size=batch_size,
                         num_workers=0,
                         shuffle=False,
                         pin_memory=True)

    np.random.seed(SHUFFLE_SEED)  # fix numpy seed to obtain the same val set every time, I know its hacky
    val_set = [b for b in val_set]
    np.random.seed()

    val_set_samples = DataLoader(val_dataset,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=False,
                                 pin_memory=True)

    val_set_samples = [s for i, s in enumerate(val_set_samples)
                       if i < num_gen_samples]

    return train_set, val_set, val_set_samples


class VocCollator:

    def __init__(self,
                 hop_length: int,
                 voc_pad: int,
                 voc_seq_len: int,
                 voc_mode: str,
                 bits: int):
        self.hop_length = hop_length
        self.voc_pad = voc_pad
        self.voc_seq_len = voc_seq_len
        self.voc_mode = voc_mode
        self.bits = bits

    def __call__(self, batch: List[Dict[str, torch.tensor]]) -> Dict[str, torch.tensor]:
        mel_win = self.voc_seq_len // self.hop_length + 2 * self.voc_pad
        max_offsets = [b['mel'].shape[-1] - 2 - (mel_win + 2 * self.voc_pad) for b in batch]
        mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        sig_offsets = [(offset + self.voc_pad) * self.hop_length for offset in mel_offsets]

        mels = [b['mel'][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, b in enumerate(batch)]

        labels = [b['x'][sig_offsets[i]:sig_offsets[i] + self.voc_seq_len + 1] for i, b in enumerate(batch)]

        mels = np.stack(mels).astype(np.float32)
        labels = np.stack(labels).astype(np.int64)

        mel = torch.tensor(mels)
        labels = torch.tensor(labels).long()

        x = labels[:, :self.voc_seq_len]
        y = labels[:, 1:]

        bits = 16 if self.voc_mode == 'MOL' else self.bits

        x = DSP.label_2_float(x.float(), bits)

        if self.voc_mode == 'MOL':
            y = DSP.label_2_float(y.float(), bits)

        return {'mel': mel, 'x': x, 'y': y}


###################################################################################
# Tacotron/TTS Dataset ############################################################
###################################################################################


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
                 path: Path,
                 dataset_ids: List[str],
                 text_dict: Dict[str, str],
                 tokenizer: Tokenizer) -> None:
        self.path = path
        self.metadata = dataset_ids
        self.text_dict = text_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> Dict[str, torch.tensor]:
        item_id = self.metadata[index]
        text = self.text_dict[item_id]
        x = self.tokenizer(text)
        mel = np.load(str(self.path/'mel'/f'{item_id}.npy'))
        mel_len = mel.shape[-1]
        return {'x': x, 'mel': mel, 'item_id': item_id,
                'mel_len': mel_len, 'x_len': len(x)}

    def __len__(self):
        return len(self.metadata)


class ForwardDataset(Dataset):

    def __init__(self,
                 path: Path,
                 dataset_ids: List[str],
                 text_dict: Dict[str, str],
                 tokenizer: Tokenizer):
        self.path = path
        self.metadata = dataset_ids
        self.text_dict = text_dict
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> Dict[str, torch.tensor]:
        item_id = self.metadata[index]
        text = self.text_dict[item_id]
        x = self.tokenizer(text)
        mel = np.load(str(self.path/'mel'/f'{item_id}.npy'))
        mel_len = mel.shape[-1]
        dur = np.load(str(self.path/'alg'/f'{item_id}.npy'))
        pitch = np.load(str(self.path/'phon_pitch'/f'{item_id}.npy'))
        energy = np.load(str(self.path/'phon_energy'/f'{item_id}.npy'))
        return {'x': x, 'mel': mel, 'item_id': item_id, 'x_len': len(x),
                'mel_len': mel_len, 'dur': dur, 'pitch': pitch, 'energy': energy}

    def __len__(self):
        return len(self.metadata)


class BinnedTacoDataLoader:
    """
    Special dataloader to allow tacotron inference on batches with equal input length per batch.
    This is used to safely generate attention matrices for extracting phoneme durations as there
    is no input padding needed.
    """

    def __init__(self,
                 data_path: Path,
                 dataset: List[Tuple[str, int]],
                 max_batch_size: int = 8) -> None:
        """
        Initializes the dataloader.

        Args:
            data_path: Main path to training data (paths.data)
            dataset: List of Tuples with (file_id, mel_length)
            max_batch_size: Maximum allowed batch size.
        """

        tokenizer = Tokenizer()
        file_id_text_lens = []
        text_dict = unpickle_binary(data_path / 'text_dict.pkl')
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
            batches = list(batchify(big_batch, batch_size=max_batch_size))
            all_batches.extend(batches)

        Random(SHUFFLE_SEED).shuffle(all_batches)
        self.all_batches = all_batches
        self.taco_dataset = TacoDataset(path=data_path, dataset_ids=dataset_ids,
                                        text_dict=text_dict, tokenizer=tokenizer)

    def __iter__(self) -> Iterator:
        for batch in self.all_batches:
            batch = [self.taco_dataset[i] for i in batch]
            batch = collate_tts(batch, r=1)
            yield batch

    def __len__(self) -> int:
        return len(self.all_batches)


def get_tts_datasets(path: Path,
                     batch_size: int,
                     r: int,
                     max_mel_len,
                     filter_attention=True,
                     filter_min_alignment=0.5,
                     filter_min_sharpness=0.9,
                     model_type='tacotron') -> Tuple[DataLoader, DataLoader]:

    tokenizer = Tokenizer()

    train_data = unpickle_binary(path/'train_dataset.pkl')
    val_data = unpickle_binary(path/'val_dataset.pkl')
    text_dict = unpickle_binary(path/'text_dict.pkl')

    train_data = filter_max_len(train_data, max_mel_len)
    val_data = filter_max_len(val_data, max_mel_len)
    train_len_original = len(train_data)

    if model_type == 'forward' and filter_attention:
        attention_score_dict = unpickle_binary(path/'att_score_dict.pkl')
        train_data = filter_bad_attentions(dataset=train_data,
                                           attention_score_dict=attention_score_dict,
                                           min_alignment=filter_min_alignment,
                                           min_sharpness=filter_min_sharpness)
        val_data = filter_bad_attentions(dataset=val_data,
                                         attention_score_dict=attention_score_dict,
                                         min_alignment=filter_min_alignment,
                                         min_sharpness=filter_min_sharpness)
        print(f'Using {len(train_data)} train files. '
              f'Filtered {train_len_original - len(train_data)} files due to bad attention!')

    train_ids, train_lens = zip(*train_data)
    val_ids, val_lens = zip(*val_data)

    if model_type == 'tacotron':
        train_dataset = TacoDataset(path=path, dataset_ids=train_ids,
                                    text_dict=text_dict, tokenizer=tokenizer)
        val_dataset = TacoDataset(path=path, dataset_ids=val_ids,
                                  text_dict=text_dict, tokenizer=tokenizer)
    elif model_type == 'forward':
        train_dataset = ForwardDataset(path=path, dataset_ids=train_ids,
                                       text_dict=text_dict, tokenizer=tokenizer)
        val_dataset = ForwardDataset(path=path, dataset_ids=val_ids,
                                     text_dict=text_dict, tokenizer=tokenizer)
    else:
        raise ValueError(f'Unknown model: {model_type}, must be either [tacotron, forward]!')

    train_sampler = BinnedLengthSampler(train_lens, batch_size, batch_size * 3)

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_tts(batch, r),
                           batch_size=batch_size,
                           sampler=train_sampler,
                           num_workers=0,
                           pin_memory=True)

    val_set = DataLoader(val_dataset,
                         collate_fn=lambda batch: collate_tts(batch, r),
                         batch_size=batch_size,
                         sampler=None,
                         num_workers=0,
                         shuffle=False,
                         pin_memory=True)

    return train_set, val_set


def get_binned_taco_dataloader(data_path: Path, max_batch_size: int = 8) -> BinnedTacoDataLoader:
    train_data = unpickle_binary(data_path / 'train_dataset.pkl')
    val_data = unpickle_binary(data_path / 'val_dataset.pkl')
    dataset = train_data + val_data
    return BinnedTacoDataLoader(data_path=data_path, dataset=dataset, max_batch_size=max_batch_size)


def collate_tts(batch: List[Dict[str, Union[str, torch.tensor]]], r: int) -> Dict[str, torch.tensor]:
    x_len = [b['x_len'] for b in batch]
    x_len = torch.tensor(x_len)
    max_x_len = max(x_len)
    text = [pad1d(b['x'], max_x_len) for b in batch]
    text = np.stack(text)
    text = torch.tensor(text).long()
    spec_lens = [b['mel_len'] for b in batch]
    max_spec_len = max(spec_lens) + 1
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r
    mel = [pad2d(b['mel'], max_spec_len) for b in batch]
    mel = np.stack(mel)
    mel = torch.tensor(mel)
    item_id = [b['item_id'] for b in batch]
    mel_lens = [b['mel_len'] for b in batch]
    mel_lens = torch.tensor(mel_lens)

    dur, pitch, energy = None, None, None
    if 'dur' in batch[0]:
        dur = [pad1d(b['dur'][:max_x_len], max_x_len) for b in batch]
        dur = np.stack(dur)
        dur = torch.tensor(dur).float()
    if 'pitch' in batch[0]:
        pitch = [pad1d(b['pitch'][:max_x_len], max_x_len) for b in batch]
        pitch = np.stack(pitch)
        pitch = torch.tensor(pitch).float()
    if 'energy' in batch[0]:
        energy = [pad1d(b['energy'][:max_x_len], max_x_len) for b in batch]
        energy = np.stack(energy)
        energy = torch.tensor(energy).float()

    return {'x': text, 'mel': mel, 'item_id': item_id, 'x_len': x_len,
            'mel_len': mel_lens, 'dur': dur, 'pitch': pitch, 'energy': energy}


def filter_max_len(dataset: List[tuple], max_mel_len: int) -> List[tuple]:
    if max_mel_len is None:
        return dataset
    return [(id, len) for id, len in dataset if len <= max_mel_len]


def filter_bad_attentions(dataset: List[tuple],
                          attention_score_dict: Dict[str, tuple],
                          min_alignment: float,
                          min_sharpness: float) -> List[tuple]:
    dataset_filtered = []
    for item_id, mel_len in dataset:
        align_score, sharp_score = attention_score_dict[item_id]
        if align_score > min_alignment \
                and sharp_score > min_sharpness:
            dataset_filtered.append((item_id, mel_len))
    return dataset_filtered


def pad1d(x, max_len) -> np.array:
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len) -> np.array:
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), constant_values=-11.5129, mode='constant')


def batchify(input: List[Any], batch_size: int) -> List[List[Any]]:
    input_len = len(input)
    output = []
    for i in range(0, input_len, batch_size):
        batch = input[i:min(i + batch_size, input_len)]
        output.append(batch)
    return output
