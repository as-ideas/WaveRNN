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
from utils.text.tokenizer import Tokenizer


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


def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)


class BaseDataet(Dataset):
    def __init__(self, files, mel_segment_len=50):
        self.files = files
        self.mel_segment_len = mel_segment_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        out_base = torch.load(file)
        mel = out_base['mel']
        mel_post = out_base['mel_post']
        if self.mel_segment_len is not None:
            mel_pad_len = self.mel_segment_len - out_base['mel'].size(-1)
            if mel_pad_len > 0:
                mel_pad = torch.full((mel.size(0), 80, mel_pad_len), fill_value=-11.5129)
                mel = torch.cat([mel, mel_pad], dim=-1)
            max_mel_start = mel.size(-1) - self.mel_segment_len
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_len
            mel = mel[:, :, mel_start:mel_end]
            mel_post = mel_post[:, :, mel_start:mel_end]

        return {'mel': mel.squeeze(0), 'mel_post': mel_post.squeeze(0)}


class StringDataset(Dataset):
    def __init__(self, strings):
        self.strings = strings
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        string = self.strings[idx]
        indices = self.tokenizer(string)
        return torch.LongTensor(indices)



if __name__ == '__main__':

    files = list(Path('/Users/cschaefe/datasets/finetuning/bild_welt_masked_mels').glob('**/*.pt'))
    dataset = BaseDataet(files)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=2)
    tts_path = '/Users/cschaefe/stream_tts_models/mm_6cons/model.pt'
    voc_path = '/Users/cschaefe/workspace/tts-synthv3/app/11111111/models/welt_voice/voc_model/model.pt'
    sw = SummaryWriter('checkpoints/logs_finetune_batched')

    val_strings = ['ɡant͡s ɔɪʁoːpa?', 'najaː, man t͡sɛɐdɛŋkt diː zaxn̩ zoː zeːɐ.', 'naɪn, fʁaʊ lampʁɛçt, diː meːdiən zɪnt nɪçt ʃʊlt.','ɛs ɪst ʃaːdə, das diː eːʔuː als diː humaːnstə ʊnt moʁaːlɪʃstə alɐ lɛndɐɡʁʊpiːʁʊŋən anɡəzeːən vɪʁt, aːbɐ ziː vɔlən diː mɛnʃn̩ʁɛçtə nɪçt aʊfʁɛçtʔɛɐhaltn̩ ʊnt deːn maɡnɪt͡ski ɛkt nɪçt nʊt͡sn̩.']
    val_dataset = StringDataset(val_strings)
    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn,
                                sampler=None)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # Instantiate Forward TTS Model
    checkpoint = torch.load(tts_path, map_location=device)
    model = MultiForwardTacotron.from_checkpoint(tts_path).to(device)
    model.eval()
    model.postnet.train()
    optimizer = optim.Adam(list(model.postnet.parameters()) + list(model.post_proj.parameters()), lr=1e-5)
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
    melgan.train()

    step = 0

    for train_batch in dataloader:

        if step % 100 == 0:
            model.eval()
            melgan.eval()
            val_loss = 0
            for i, batch in enumerate(val_dataloader):
                batch = batch.to(device)
                with torch.no_grad():
                    ada = model.generate(batch, checkpoint['speaker_embeddings']['welt'].to(device), series_transformer=series_transformer)['mel_post']
                    audio = melgan(ada)
                    audio = audio.squeeze(1)

                    audio_mel = mel_spectrogram(audio, n_fft=1024, num_mels=80,
                                                sampling_rate=22050, hop_size=256, fmin=0, fmax=8000,
                                                win_size=1024)

                    loss_exp = torch.norm(torch.exp(audio_mel) - torch.exp(ada), p="fro") / torch.norm(torch.exp(ada), p="fro") * 10.
                    val_loss += loss_exp.item()

                    loss_time = (torch.exp(audio_mel) - torch.exp(ada)) ** 2
                    loss_time = 10 * loss_time
                    loss_time = loss_time.mean(dim=1)[0]

                    time_fig = plot_pitch(loss_time.detach().cpu().numpy())
                    sw.add_figure(f'mel_loss_time_{i}/val', time_fig, global_step=step)

                    audio_inf = melgan.inference(ada)

                    mel_plot = plot_mel(audio_mel.squeeze().detach().cpu().numpy()[:100, :100])
                    mel_plot_target = plot_mel(ada.squeeze().detach().cpu().numpy()[:100, :100])
                    mel_plot_ada = plot_mel(ada.squeeze().detach().cpu().numpy()[:100, :100])
                    mel_diff = (audio_mel - ada).squeeze().detach().cpu().numpy()[:100, :100]
                    mel_exp_diff = (torch.exp(audio_mel) - torch.exp(ada)).squeeze().detach().cpu().numpy()[:100, :100]
                    mel_diff_plot = plot_mel(mel_diff)
                    mel_exp_diff_plot = plot_mel(mel_exp_diff)
                    sw.add_audio(f'audio_generated_{i}', audio_inf.detach().cpu(), sample_rate=22050, global_step=step)
                    sw.add_figure(f'generated_{i}', mel_plot, global_step=step)
                    sw.add_figure(f'target_tts_{i}', mel_plot_target, global_step=step)
                    sw.add_figure(f'ada_tts_{i}', mel_plot_ada, global_step=step)
                    sw.add_figure(f'diff_tts_{i}', mel_diff_plot, global_step=step)
                    sw.add_figure(f'diff_exp_tts_{i}', mel_exp_diff_plot, global_step=step)
            sw.add_scalar('mel_loss/val', val_loss / len(val_dataloader), global_step=step)
            checkpoint['model'] = model.state_dict()
            torch.save(checkpoint, 'checkpoints/forward_taco_finetuned_post_new.pt')
            model.postnet.rnn.train()
            model.post_proj.train()

        batch = {'mel': train_batch['mel'].to(device), 'mel_post': train_batch['mel_post'].to(device)}
        ada = model.postnet(batch['mel'])
        ada = model.post_proj(ada).transpose(1, 2)

        audio = melgan(ada)
        audio = audio.squeeze(1)
        audio_mel = mel_spectrogram(audio, n_fft=1024, num_mels=80,
                                    sampling_rate=22050, hop_size=256, fmin=0, fmax=8000,
                                    win_size=1024)

        audio_mel[batch['mel_post'] < -11] = -11.51
        ada[batch['mel_post'] < -11] = -11.51
        batch['mel_post'][batch['mel_post'] < -11] = -11.51

        loss_exp = torch.norm(torch.exp(audio_mel) - torch.exp(batch['mel_post']), p="fro") / torch.norm(torch.exp(batch['mel_post']), p="fro") * 10.
        loss_log = F.l1_loss(ada, batch['mel_post'])

        loss_tot = (loss_exp + loss_log)
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        step += 1

        sw.add_scalar('mel_exp_loss/train', loss_exp, global_step=step)
        sw.add_scalar('mel_log_loss/train', loss_log, global_step=step)

        print(step, loss_exp, loss_log)

