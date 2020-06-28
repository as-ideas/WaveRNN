from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
from utils.dsp import *
from utils import hparams as hp
from utils.files import unpickle_binary
from utils.text import text_to_sequence
from pathlib import Path
import random


###################################################################################
# WaveRNN/Vocoder Dataset #########################################################
###################################################################################


class VocoderDataset(Dataset):
    def __init__(self, path: Path, dataset_ids, train_gta=False):
        self.metadata = dataset_ids
        self.mel_path = path/'gta' if train_gta else path/'mel'
        self.quant_path = path/'quant'


    def __getitem__(self, index):
        item_id = self.metadata[index]
        m = np.load(self.mel_path/f'{item_id}.npy')
        x = np.load(self.quant_path/f'{item_id}.npy')
        return m, x

    def __len__(self):
        return len(self.metadata)


def get_vocoder_datasets(path: Path, batch_size, train_gta):
    train_data = unpickle_binary(path/'train_dataset.pkl')
    val_data = unpickle_binary(path/'val_dataset.pkl')
    train_ids, train_lens = filter_max_len(train_data)
    val_ids, val_lens = filter_max_len(val_data)

    train_dataset = VocoderDataset(path, train_ids, train_gta)
    val_dataset = VocoderDataset(path, val_ids, train_gta)

    train_set = DataLoader(train_dataset,
                           collate_fn=collate_vocoder,
                           batch_size=batch_size,
                           num_workers=2,
                           shuffle=True,
                           pin_memory=True)

    val_set = DataLoader(val_dataset,
                         collate_fn=collate_vocoder,
                         batch_size=batch_size,
                         num_workers=1,
                         shuffle=False,
                         pin_memory=True)

    np.random.seed(42)  # fix numpy seed to obtain the same val set every time, I know its hacky
    val_set = [b for b in val_set]
    np.random.seed()

    val_set_samples = DataLoader(val_dataset,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False,
                                 pin_memory=True)

    val_set_samples = [s for i, s in enumerate(val_set_samples)
                       if i < hp.voc_gen_num_samples]

    return train_set, val_set, val_set_samples


def collate_vocoder(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits

    x = label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL':
        y = label_2_float(y.float(), bits)

    return x, y, mels


###################################################################################
# Tacotron/TTS Dataset ############################################################
###################################################################################


def get_tts_datasets(path: Path, batch_size, r, model_type='tacotron'):
    train_data = unpickle_binary(path/'train_dataset.pkl')
    val_data = unpickle_binary(path/'val_dataset.pkl')
    train_ids, train_lens = filter_max_len(train_data)
    val_ids, val_lens = filter_max_len(val_data)
    text_dict = unpickle_binary(path/'text_dict.pkl')
    if model_type == 'tacotron':
        train_dataset = TacoDataset(path, train_ids, text_dict)
        val_dataset = TacoDataset(path, val_ids, text_dict)
    elif model_type == 'forward':
        train_dataset = ForwardDataset(path, train_ids, text_dict)
        val_dataset = ForwardDataset(path, val_ids, text_dict)
    else:
        raise ValueError(f'Unknown model: {model_type}, must be either [tacotron, forward]!')

    train_sampler = BinnedLengthSampler(train_lens, batch_size, batch_size * 3)

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_tts(batch, r),
                           batch_size=batch_size,
                           sampler=train_sampler,
                           num_workers=1,
                           pin_memory=True)

    val_set = DataLoader(val_dataset,
                         collate_fn=lambda batch: collate_tts(batch, r),
                         batch_size=batch_size,
                         sampler=None,
                         num_workers=1,
                         shuffle=False,
                         pin_memory=True)

    return train_set, val_set


def filter_max_len(dataset):
    dataset_ids = []
    mel_lengths = []
    for item_id, mel_len in dataset:
        if mel_len <= hp.tts_max_mel_len:
            dataset_ids += [item_id]
            mel_lengths += [mel_len]
    return dataset_ids, mel_lengths
    

class TacoDataset(Dataset):

    def __init__(self, path: Path, dataset_ids, text_dict):
        self.path = path
        self.metadata = dataset_ids
        self.text_dict = text_dict

    def __getitem__(self, index):
        item_id = self.metadata[index]
        text = self.text_dict[item_id]
        x = text_to_sequence(text)
        mel = np.load(self.path/'mel'/f'{item_id}.npy')
        mel_len = mel.shape[-1]
        return x, mel, item_id, mel_len

    def __len__(self):
        return len(self.metadata)


class ForwardDataset(Dataset):

    def __init__(self, path: Path, dataset_ids, text_dict):
        self.path = path
        self.metadata = dataset_ids
        self.text_dict = text_dict

    def __getitem__(self, index):
        item_id = self.metadata[index]
        text = self.text_dict[item_id]
        x = text_to_sequence(text)
        mel = np.load(self.path/'mel'/f'{item_id}.npy')
        mel_len = mel.shape[-1]
        dur = np.load(self.path/'alg'/f'{item_id}.npy')
        return x, mel, item_id, mel_len, dur

    def __len__(self):
        return len(self.metadata)


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant')


def collate_tts(batch, r):
    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)
    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)
    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r
    mel = [pad2d(x[1], max_spec_len) for x in batch]
    mel = np.stack(mel)
    ids = [x[2] for x in batch]
    mel_lens = [x[3] for x in batch]
    mel_lens = torch.tensor(mel_lens)
    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    # scale spectrograms to -4 <--> 4
    mel = (mel * 8.) - 4.
    # additional durations for forward
    if len(batch[0]) > 4:
        dur = [pad1d(x[4][:max_x_len], max_x_len) for x in batch]
        dur = np.stack(dur)
        dur = torch.tensor(dur).float()
        return chars, mel, ids, mel_lens, dur
    else:
        return chars, mel, ids, mel_lens


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
