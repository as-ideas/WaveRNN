import itertools
import itertools
import random
from pathlib import Path
from random import Random
from typing import List
from typing import Union

import pandas as pd
import ruamel.yaml
import torch
import torch.nn.functional as F
# Define the dataset
import tqdm
from dp.phonemizer import Phonemizer
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch import optim
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from melgan import Generator
from models.fast_pitch import FastPitch
from models.forward_tacotron import ForwardTacotron, ForwardSeriesTransformer
from models.multi_forward_tacotron import MultiForwardTacotron
from trainer.common import to_device
from utils.display import *
from utils.text.symbols import phonemes


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

import torch
import torch.nn as nn
import torch.nn.init as init


class ConvFilter(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the convolutional layer
        self.conv = nn.Conv2d(1, 1, (5, 1), 1, (2, 0))
        # Initialize the weights of the convolutional layer to 1
        init.constant_(self.conv.weight, 1)
        # Initialize the biases of
        self.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def create_gta_features(model: Union[ForwardTacotron, FastPitch],
                        train_set: DataLoader,
                        val_set: DataLoader,
                        save_path: Path) -> None:
    model.eval()
    device = next(model.parameters()).device  # use same device as model parameters
    iters = len(train_set) + len(val_set)
    dataset = itertools.chain(train_set, val_set)
    for i, batch in enumerate(dataset, 1):
        batch = to_device(batch, device=device)
        with torch.no_grad():
            pred = model(batch)
        gta = pred['mel_post'].cpu().numpy()
        for j, item_id in enumerate(batch['item_id']):
            mel = gta[j][:, :batch['mel_len'][j]]
            np.save(str(save_path/f'{item_id}.npy'), mel, allow_pickle=False)
        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


class Tokenizer:

    def __init__(self) -> None:
        self.symbol_to_id = {s: i for i, s in enumerate(phonemes + ['|'])}
        self.id_to_symbol = {i: s for i, s in enumerate(phonemes + ['|'])}

    def __call__(self, text: str) -> List[int]:
        return [self.symbol_to_id[t] for t in text if t in self.symbol_to_id]

    def decode(self, sequence: List[int]) -> str:
        text = [self.id_to_symbol[s] for s in sequence if s in self.id_to_symbol]
        return ''.join(text)

def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)


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


class StringDataset(Dataset):
    def __init__(self, strings, phonemizer=None):
        self.strings = strings
        self.tokenizer = Tokenizer()
        self.phonemizer = phonemizer

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        string = self.strings[idx]
        if self.phonemizer is not None:
            string = self.phonemizer(string, lang='de')
            if 'θ' in string:
                print(string)
        indices = self.tokenizer(string)
        return torch.LongTensor(indices)










if __name__ == '__main__':

    tts_path = '/Users/cschaefe/stream_tts_models/bild_welt_masked_welt/model.pt'
    voc_path = '/Users/cschaefe/workspace/tts-synthv3/app/11111111/models/welt_voice/voc_model/model.pt'
    speaker = 'welt'
    df = pd.read_csv('/Users/cschaefe/datasets/finetuning/metadata_processed.tsv', sep='\t', encoding='utf-8')

    #tts_path = 'welt_voice/tts_model/model.pt'
    #voc_path = 'welt_voice/voc_model/model.pt'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # Instantiate Forward TTS Model
    checkpoint = torch.load(tts_path, map_location=device)
    model = MultiForwardTacotron.from_checkpoint(tts_path).to(device)
    model_base = MultiForwardTacotron.from_checkpoint(tts_path).to(device)

    speed_factor, pitch_factor = 1., 1.
    phoneme_min_duration = {}
    if (Path(tts_path).parent/'config.yaml').exists():
        with open(str(Path(tts_path).parent/'config.yaml'), 'rb') as data_yaml:
            config = ruamel.yaml.YAML().load(data_yaml)
            speed_factor = config.get('speed_factor', 1)
            pitch_factor = config.get('pitch_factor', 1)
            print('pitch factor: ', pitch_factor)
            phoneme_min_duration = config.get('phoneme_min_duration', {})

    series_transformer = ForwardSeriesTransformer(tokenizer=Tokenizer(),
                                                  phoneme_min_duration=phoneme_min_duration,
                                                  speed_factor=speed_factor,
                                                  pitch_factor=pitch_factor)

    melgan = Generator(80)
    voc_checkpoint = torch.load(voc_path, map_location=torch.device('cpu'))
    melgan.load_state_dict(voc_checkpoint['model_g'])
    melgan = melgan.to(device)
    model = model.to(device)
    model_base.eval()
    melgan.eval()

    print(f'\nInitialized tts model: {model}\n')

    optimizer = optim.Adam(list(model.postnet.rnn.parameters()) + list(model.post_proj.parameters()), lr=1e-5)
    strings = df['text_phonemized']
    strings = [s for s in strings if len(s) > 10 and len(s) < 1000]
    random = Random(42)
    random.shuffle(strings)

    dataset = StringDataset(strings)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                            sampler=None)

    index = 0
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        batch = batch.to(device)
        with torch.no_grad():
            out_base = model_base.generate(batch, checkpoint['speaker_embeddings'][speaker].to(device), series_transformer=series_transformer)
            audio = melgan(out_base['mel_post'])
            audio = audio.squeeze(1)
            audio_mel = mel_spectrogram(audio, n_fft=1024, num_mels=80,
                                        sampling_rate=22050, hop_size=256, fmin=0, fmax=8000,
                                        win_size=1024)
            diff = (torch.exp(audio_mel) - torch.exp(out_base['mel_post'])) ** 2
            diff = diff.mean(1)
            diff[diff < 0.005] = 0

            ind = int(torch.argmax(diff))
            val = float(diff[0, ind])

            if val > 0:
                left = max(0, ind-32)
                right = min(left + 32, out_base['mel_post'].size(-1))
                piece = out_base['mel_post'][:, left:right]
                score = 100 * val
                torch.save({'mel_post': piece}, f'/Users/cschaefe/datasets/finetuning/bild_welt_masked_mels/{index}_{score:#.4}.pt')
                index += 1

