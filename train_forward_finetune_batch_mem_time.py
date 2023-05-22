import random
from pathlib import Path

import random
from pathlib import Path
from random import Random
from typing import Union

import ruamel.yaml
import torch.nn.functional as F
# Define the dataset
import tqdm
from librosa.filters import mel as librosa_mel_fn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from melgan import Generator
from models.forward_tacotron import ForwardSeriesTransformer
from models.multi_forward_tacotron import MultiForwardTacotron
from utils.display import *
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


class BaseDataset(Dataset):
    def __init__(self, files, mel_segment_len: Union[int, None] = 64):
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
                mel_pad = torch.full((1, 80, mel_pad_len), fill_value=-11.5129).to(mel.device)
                mel = torch.cat([mel, mel_pad], dim=-1)
                mel_post = torch.cat([mel_post, mel_pad], dim=-1)
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

    files = list(Path('/Users/cschaefe/datasets/finetuning/bild_welt_masked_mels_eval').glob('**/*.pt'))

    Random(42).shuffle(files)

    n_val = 2
    val_files = files[:n_val]
    train_files = files[n_val:]

    mel_segment_len = 128
    mel_segment_len_2 = 64

    dataset = BaseDataset(train_files, mel_segment_len=mel_segment_len)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=0)
    val_dataset = BaseDataset(val_files, mel_segment_len=None)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0)

    tts_path = '/Users/cschaefe/stream_tts_models/bild_welt_masked_welt/model.pt'
    voc_path = '/Users/cschaefe/workspace/tts-synthv3/app/11111111/models/welt_voice/voc_model/model.pt'
    sw = SummaryWriter('checkpoints/logs_finetune_batched')
    save_path = 'checkpoints/finetuning/forward_taco_finebatch'

    val_strings = ['ɡant͡s ɔɪʁoːpa?',
                   'çiːna bəkɛmp͡ft diː koʁoːna-pandemiː fɔn bəɡɪn an mɪt aɪnəm ʊltʁa-ʃtʁɛŋən nəʊ-kovɪt-ʁeʒiːm.',
                   'najaː, man t͡sɛɐdɛŋkt diː zaxn̩ zoː zeːɐ.',
                   'naɪn, fʁaʊ lampʁɛçt, diː meːdiən zɪnt nɪçt ʃʊlt.',
                   'ɛs ɪst ʃaːdə, das diː eːʔuː als diː humaːnstə ʊnt moʁaːlɪʃstə alɐ lɛndɐɡʁʊpiːʁʊŋən anɡəzeːən vɪʁt, aːbɐ ziː vɔlən diː mɛnʃn̩ʁɛçtə nɪçt aʊfʁɛçtʔɛɐhaltn̩ ʊnt deːn maɡnɪt͡ski ɛkt nɪçt nʊt͡sn̩.',
                   'ɪn aɪnɐ vɛlt, ɪn deːɐ deːɐ kliːmavandl̩ ɪmɐ dʁɛŋəndɐ vɪʁt, hat diː ɔɪʁopɛːɪʃə unioːn aɪnən eːɐɡaɪt͡sɪɡn̩ plaːn foːɐɡəʃlaːɡn̩: bɪs t͡svaɪtaʊzn̩t-fʏnfʊntdʁaɪsɪç zɔlən kaɪnə aʊtos meːɐ aʊf deːn maʁkt kɔmən, diː t͡seː-ʔoː- t͡svaɪ oːdɐ andəʁə ʃaːtʃtɔfə aʊsʃtoːsn̩.']

    plot_dataset = StringDataset(val_strings)
    plot_dataloader = DataLoader(plot_dataset, batch_size=1, collate_fn=collate_fn,
                                 sampler=None)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # Instantiate Forward TTS Model
    checkpoint = torch.load(tts_path, map_location=device)
    model = MultiForwardTacotron.from_checkpoint(tts_path).to(device)
    model_base = MultiForwardTacotron.from_checkpoint(tts_path).to(device)
    model_base.eval()
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
    melgan.eval()

    step = 0

    for epoch in range(1000):
        for train_batch in dataloader:

            if step % 100 == 0:
                model.eval()
                val_loss_exp = 0
                val_loss_log = 0

                for i, batch in tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                    with torch.no_grad():
                        batch = {'mel': batch['mel'].to(device), 'mel_post': batch['mel_post'].to(device)}
                        ada = model.postnet(batch['mel'])
                        ada = model.post_proj(ada).transpose(1, 2)
                        audio = melgan(ada)
                        audio = audio.squeeze(1)

                        audio_mel = mel_spectrogram(audio, n_fft=1024, num_mels=80,
                                                    sampling_rate=22050, hop_size=256, fmin=0, fmax=8000,
                                                    win_size=1024)

                        #loss_exp = torch.norm(torch.exp(audio_mel) - torch.exp(batch['mel_post']), p="fro") / torch.norm(torch.exp(batch['mel_post']), p="fro") * 10.
                        diff = (torch.exp(audio_mel) - torch.exp(batch['mel_post'])) ** 2
                        diff = diff.mean(1)
                        diff[diff < 0.005] = 0
                        loss_exp = 10. * diff.mean()

                        loss_log = F.l1_loss(ada, batch['mel_post'])
                        val_loss_exp += loss_exp.item()
                        val_loss_log += loss_log.item()

                sw.add_scalar('mel_loss_exp/val', val_loss_exp / len(val_dataloader), global_step=step)
                sw.add_scalar('mel_loss_log/val', val_loss_log / len(val_dataloader), global_step=step)
                checkpoint['model'] = model.state_dict()
                k_steps = (step // 10000) * 10
                torch.save(checkpoint, f'{save_path}_{k_steps}k.pt')

                for i, batch in enumerate(plot_dataloader):
                    batch = batch.to(device)
                    with torch.no_grad():
                        ada = model.generate(batch, checkpoint['speaker_embeddings']['welt'].to(device), series_transformer=series_transformer)['mel_post']
                        mel_base = model_base.generate(batch, checkpoint['speaker_embeddings']['welt'].to(device), series_transformer=series_transformer)['mel_post']
                        audio = melgan(ada)
                        audio = audio.squeeze(1)
                        audio_mel = mel_spectrogram(audio, n_fft=1024, num_mels=80,
                                                    sampling_rate=22050, hop_size=256, fmin=0, fmax=8000,
                                                    win_size=1024)
                        loss_time = (torch.exp(audio_mel) - torch.exp(mel_base)) ** 2
                        loss_time = 10 * loss_time.mean(dim=1)[0]
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
                model.postnet.train()
                model.post_proj.train()

            batch = {'mel': train_batch['mel'].to(device), 'mel_post': train_batch['mel_post'].to(device)}
            ada = model.postnet(batch['mel'])
            ada = model.post_proj(ada).transpose(1, 2)

            max_mel_start = ada.size(-1) - mel_segment_len_2
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + mel_segment_len_2

            ada = ada[:, :, mel_start:mel_end]
            batch['mel_post'] = batch['mel_post'][:, :, mel_start:mel_end]

            audio = melgan(ada)
            audio = audio.squeeze(1)
            audio_mel = mel_spectrogram(audio, n_fft=1024, num_mels=80,
                                        sampling_rate=22050, hop_size=256, fmin=0, fmax=8000,
                                        win_size=1024)

            #loss_exp = torch.norm(torch.exp(audio_mel) - torch.exp(batch['mel_post']), p="fro") / torch.norm(torch.exp(batch['mel_post']), p="fro") * 10.
            diff = (torch.exp(audio_mel) - torch.exp(batch['mel_post'])) ** 2
            diff = diff.mean(1)
            diff[diff < 0.005] = 0
            loss_exp = 10. * diff.sum()

            loss_log = F.l1_loss(ada, batch['mel_post'])

            loss_tot = (loss_exp + loss_log)
            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()

            step += 1

            sw.add_scalar('mel_exp_loss/train', loss_exp, global_step=step)
            sw.add_scalar('mel_log_loss/train', loss_log, global_step=step)

            print(step, loss_exp, loss_log)

