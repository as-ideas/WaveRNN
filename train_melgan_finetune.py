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

    val_strings = ['najaː, man t͡sɛɐdɛŋkt diː zaxn̩ zoː zeːɐ.', 'naɪn, fʁaʊ lampʁɛçt, diː meːdiən zɪnt nɪçt ʃʊlt.', 'diː t͡seː-deː-ʔuː-t͡sɛntʁaːlə lɛst iːɐ ʃpɪt͡sn̩pɛʁzonaːl dʊʁçt͡ʃɛkn̩: ɪm jʏŋstn̩ mɪtɡliːdɐbʁiːf fɔn bʊndəsɡəʃɛft͡sfyːʁɐ ʃtɛfan hɛnəvɪç (axtʔʊntfɪʁt͡sɪç) vɪʁt diː t͡seː-deː-ʔuː-baːzɪs aʊfɡəfɔʁdɐt, an aɪnɐ bəfʁaːɡʊŋ dɛs tʁiːʁɐ paʁtaɪən fɔʁʃɐs uːvə jan (nɔɪnʔʊntfʏnft͡sɪç) taɪlt͡suneːmən.']

    tts_path = '/Users/cschaefe/workspace/tts-synthv3/app/11111111/models/welt_voice/tts_model/model.pt'
    voc_path = '/Users/cschaefe/workspace/tts-synthv3/app/11111111/models/welt_voice/voc_model/model.pt'
    #tts_path = 'welt_voice/tts_model/model.pt'
    #voc_path = 'welt_voice/voc_model/model.pt'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # Instantiate Forward TTS Model
    model = ForwardTacotron.from_checkpoint(tts_path)
    model_base = ForwardTacotron.from_checkpoint(tts_path).to(device)

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
    model.postnet.train()
    model.post_proj.train()
    model_base.eval()
    melgan.eval()
    print(f'\nInitialized tts model: {model}\n')

    class Adapter(nn.Module):
        def __init__(self):
            super(Adapter, self).__init__()
            self.conv = nn.Conv1d(80, 256, 3, padding=1)
            self.conv_2 = nn.Conv1d(256, 80, 3, padding=1)
            #self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
            #self.lin = nn.Linear(512, 80)

        def forward(self, x):
            x = self.conv(x)
            # x, _ = self.gru(x.transpose(1, 2))
            # x = self.lin(x)
            #x = x.transpose(1, 2)
            x = self.conv_2(x)
            return x

    #adapter = Sequential(
    #    weight_norm(nn.Conv1d(80, 512, 5, padding=2)),
    #    weight_norm(nn.Conv1d(512, 80, 5, padding=2)),
    #)
    adapter = Adapter()
    optimizer = optim.Adam(list(model.postnet.rnn.parameters()) + list(model.post_proj.parameters()), lr=1e-5)
    phonemizer = Phonemizer.from_checkpoint('/Users/cschaefe/workspace/tts-synthv3/app/11111111/models/welt_voice/phon_model/model.pt')
    #df = pd.read_csv('/Users/cschaefe/datasets/nlp/welt_articles_phonemes.tsv', sep='\t', encoding='utf-8')
    df = pd.read_csv('/Users/cschaefe/datasets/tts-synth-data/welt/processed_metadata.tsv', sep='\t', encoding='utf-8')
    df.dropna(inplace=True)
    strings = df['text_phonemized']
    strings = [s for s in strings if len(s) > 10 and len(s) < 300]
    random = Random(42)
    random.shuffle(strings)

    dataset = StringDataset(strings)
    val_dataset = StringDataset(val_strings)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                            sampler=None)

    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn,
                                sampler=None)

    sw = SummaryWriter('checkpoints/logs_finetune_postnet_l2_local')
    checkpoint = torch.load(tts_path, map_location=torch.device('cpu'))
    step = 0
    loss_acc = 1
    loss_sum = 0

    conv_filter = ConvFilter().to(device)

    for epoch in range(10000):
        for batch in tqdm.tqdm(dataloader, total=len(dataloader)):

            if step % 100 == 0:
                model.eval()
                melgan.eval()
                val_loss = 0
                for i, batch in enumerate(val_dataloader):
                    batch = batch.to(device)
                    with torch.no_grad():
                        out_base = model_base.generate(batch, series_transformer)
                        ada = model.postnet(out_base['mel'])
                        ada = model.post_proj(ada).transpose(1, 2)
                        audio = melgan(ada)
                        audio = audio.squeeze(1)
                        audio_mel = mel_spectrogram(audio, n_fft=1024, num_mels=80,
                                                    sampling_rate=22050, hop_size=256, fmin=0, fmax=8000,
                                                    win_size=1024)


                        loss = F.l1_loss(audio_mel, out_base['mel_post'])
                        val_loss += loss.item()

                        #torch.save(audio_mel, '/Users/cschaefe/datasets/welt_fuckup/fuckup.pt')
                        #torch.save(out_base['mel_post'], '/Users/cschaefe/datasets/fuckup_postnet.pt')

                        mel_mask = audio_mel > -1


                        loss_time = torch.exp(audio_mel) - torch.exp(out_base['mel_post'])
                        loss_time = 10 * loss_time
                        loss_time = loss_time.mean(dim=1)[0]
                        print(step, i, loss_time)
                        time_fig = plot_pitch(loss_time.detach().cpu().numpy())
                        sw.add_figure(f'mel_loss_time_{i}/val', time_fig, global_step=step)

                        audio_inf = melgan.inference(ada)
                        #audio_inf = audio_inf.squeeze(1)

                        mel_plot = plot_mel(audio_mel.squeeze().detach().cpu().numpy()[:100, :100])
                        mel_plot_target = plot_mel(out_base['mel_post'].squeeze().detach().cpu().numpy()[:100, :100])
                        mel_diff = (audio_mel - out_base['mel_post']).squeeze().detach().cpu().numpy()[:100, :100]
                        mel_exp_diff = (torch.exp(audio_mel) - torch.exp(out_base['mel_post'])).squeeze().detach().cpu().numpy()[:100, :100]
                        mel_diff_plot = plot_mel(mel_diff)
                        diff_filt = conv_filter(torch.exp(audio_mel) - torch.exp(out_base['mel_post']))
                        mel_exp_diff_conv = (diff_filt).squeeze().detach().cpu().numpy()[:100, :100]



                        mel_exp_diff_plot = plot_mel(mel_exp_diff)
                        mel_exp_diff_conv_plot = plot_mel(mel_exp_diff_conv)
                        sw.add_audio(f'audio_generated_{i}', audio_inf.detach().cpu(), sample_rate=22050, global_step=step)
                        with torch.no_grad():
                            audio_base = melgan(out_base['mel']).squeeze(1)
                            audio_base_mel = mel_spectrogram(audio_base, n_fft=1024, num_mels=80,
                                                        sampling_rate=22050, hop_size=256, fmin=0, fmax=8000,
                                                        win_size=1024)
                            mel_plot_target_base = plot_mel(audio_base_mel.squeeze().detach().cpu().numpy())

                        sw.add_audio(f'audio_target_{i}', audio_base.detach().cpu(), sample_rate=22050, global_step=step)

                        sw.add_figure(f'generated_{i}', mel_plot, global_step=step)
                        sw.add_figure(f'target_tts_{i}', mel_plot_target, global_step=step)
                        sw.add_figure(f'diff_tts_{i}', mel_diff_plot, global_step=step)
                        sw.add_figure(f'diff_exp_tts_{i}', mel_exp_diff_plot, global_step=step)
                        sw.add_figure(f'diff_exp_conv_tts_{i}', mel_exp_diff_conv_plot, global_step=step)
                sw.add_scalar('mel_loss/val', val_loss / len(val_dataloader), global_step=step)
                checkpoint['model'] = model.state_dict()
                torch.save(checkpoint, 'checkpoints/forward_taco_finetuned_post_new.pt')
                model.postnet.rnn.train()
                model.post_proj.train()

            batch = batch.to(device)
            with torch.no_grad():
                out_base = model_base.generate(batch, series_transformer=series_transformer)

            ada = model.postnet(out_base['mel'])
            ada = model.post_proj(ada).transpose(1, 2)
            audio = melgan(ada)
            audio = audio.squeeze(1)

            audio_mel = mel_spectrogram(audio, n_fft=1024, num_mels=80,
                                        sampling_rate=22050, hop_size=256, fmin=0, fmax=8000,
                                        win_size=1024)

            #loss = F.l1_loss(audio_mel, out_base['mel_post']) * 10.
            loss_log = F.l1_loss(ada, out_base['mel_post'])
            mel_mask = audio_mel > -1

            #loss_exp = F.mse_loss(torch.exp(audio_mel), torch.exp(out_base['mel_post'])) * 1000.
            #loss_exp = F.mse_loss(torch.exp(audio_mel[mel_mask]), torch.exp(out_base['mel_post'][mel_mask])) / (mel_mask.sum() + 1e-9) * 100000.
            loss_exp = torch.norm(torch.exp(audio_mel) - torch.exp(out_base['mel_post']), p="fro") / torch.norm(torch.exp(out_base['mel_post']), p="fro") * 10.

            #print(step, loss)
            #print(step, loss_log)
            print(step, loss_exp)

            #loss_time = F.l1_loss(audio_mel, out_base['mel_post'], reduction='none')
            #loss_time = loss_time.mean(dim=-1)
            #print(loss_time)
            #fig = plot_pitch(loss_time.detach().cpu().numpy())
            loss_tot = (loss_exp + loss_log)
            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()


            sw.add_scalar('mel_exp_loss/train', loss_exp, global_step=step)
            #sw.add_scalar('mel_loss/train', loss, global_step=step)
            sw.add_scalar('mel_log_loss/train', loss_log, global_step=step)


            step += 1



