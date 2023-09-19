import time
from typing import Dict, Any, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import get_window
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from distutils.version import LooseVersion
from discriminator import MultiScaleDiscriminator
from melgan import Generator
from models.multi_fast_pitch import MultiFastPitch
from models.multi_forward_tacotron import MultiForwardTacotron
from trainer.common import Averager, TTSSession, MaskedL1, to_device, np_now
from utils.checkpoints import save_checkpoint
from utils.dataset import get_forward_dataloaders
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel, plot_pitch
from utils.dsp import DSP
from utils.files import parse_schedule, unpickle_binary
from utils.paths import Paths


import tqdm
from librosa.filters import mel as librosa_mel_fn
import librosa
import torch
import numpy as np
from librosa.util import normalize

from torch.utils.data import Dataset, DataLoader


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


is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")

def stft(x: torch.Tensor,
         n_fft: int,
         hop_length: int,
         win_length: int) -> torch.Tensor:
    window = torch.hann_window(win_length, device=x.device)
    if is_pytorch_17plus:
        x_stft = torch.stft(
            input=x, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, return_complex=False)
    else:
        x_stft = torch.stft(
            input=x, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window)

    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(input_data.device),
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window.to(magnitude.device))

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return

class MultiResStftLoss(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.n_ffts = [1024, 2048, 512]
        self.hop_sizes = [120, 240, 50]
        self.win_lengths = [600, 1200, 240]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_loss = 0.
        spec_loss = 0.
        for n_fft, hop_length, win_length in zip(self.n_ffts, self.hop_sizes, self.win_lengths):
            x_stft = stft(x=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            y_stft = stft(x=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            norm_loss += F.l1_loss(torch.log(x_stft), torch.log(y_stft))
            spec_loss += torch.norm(y_stft - x_stft, p="fro") / torch.norm(y_stft, p="fro")
        return norm_loss / len(self.n_ffts), spec_loss / len(self.n_ffts)

class MultiForwardTrainer:

    def __init__(self,
                 paths: Paths,
                 dsp: DSP,
                 config: Dict[str, Any]) -> None:
        self.paths = paths
        self.dsp = dsp
        self.config = config
        self.train_cfg = config[config['tts_model']]['training']
        self.writer = SummaryWriter(log_dir=paths.forward_log, comment='v1')
        self.l1_loss = MaskedL1()
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.speakers = sorted(list(set(unpickle_binary(paths.data / 'speaker_dict.pkl').values())))
        self.speaker_embs = {}
        for speaker in self.speakers:
            speaker_emb = np.load(paths.mean_speaker_emb / f'{speaker}.npy')
            speaker_emb = torch.from_numpy(speaker_emb).float().unsqueeze(0)
            self.speaker_embs[speaker] = speaker_emb

    def train(self,
              model: Union[MultiForwardTacotron, MultiFastPitch],
              optimizer: Optimizer) -> None:
        forward_schedule = self.train_cfg['schedule']
        forward_schedule = parse_schedule(forward_schedule)
        for i, session_params in enumerate(forward_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                filter_params = self.train_cfg['filter']
                train_set, val_set = get_forward_dataloaders(
                    paths=self.paths, batch_size=bs, **filter_params)
                session = TTSSession(
                    index=i, r=1, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self,  model: Union[MultiForwardTacotron, MultiFastPitch],
                      optimizer: Optimizer, session: TTSSession) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        simple_table([(f'Steps', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr)])

        for g in optimizer.param_groups:
            g['lr'] = session.lr

        averages = {'mel_loss': Averager(), 'dur_loss': Averager(), 'step_duration': Averager()}
        device = next(model.parameters()).device  # use same device as model parameters

        multires_stft_loss = MultiResStftLoss().to(device)
        g_model = Generator(1024).to(device)
        d_model = MultiScaleDiscriminator().to(device)

        #voc_path = '/Users/cschaefe/stream_tts_models/melgan_welt_hifi_istft/model.pt'
        #voc_checkpoint = torch.load(voc_path, map_location=torch.device('cpu'))
        #g_model.load_state_dict(voc_checkpoint['model_g'])
        #d_model.load_state_dict(voc_checkpoint['model_d'])
        g_optim = Adam(g_model.parameters(), lr=2e-5,  betas=(0.5, 0.9))
        d_optim = Adam(d_model.parameters(), lr=2e-5,  betas=(0.5, 0.9))
        #d_optim.load_state_dict(voc_checkpoint['optim_d'])

        torch_stft = TorchSTFT(filter_length=16, hop_length=4, win_length=16).to(device)

        for e in range(1, epochs + 1):
            for i, batch in enumerate(session.train_set, 1):

                batch = to_device(batch, device=device)
                start = time.time()
                model.train()

                pitch_target = batch['pitch'].detach().clone()
                energy_target = batch['energy'].detach().clone()

                pred = model(batch)

                m1_loss = self.l1_loss(pred['mel'], batch['mel'], batch['mel_len'])
                m2_loss = self.l1_loss(pred['mel_post'], batch['mel'], batch['mel_len'])

                dur_loss = self.l1_loss(pred['dur'].unsqueeze(1), batch['dur'].unsqueeze(1), batch['x_len'])
                pitch_loss = self.l1_loss(pred['pitch'], pitch_target.unsqueeze(1), batch['x_len'])
                energy_loss = self.l1_loss(pred['energy'], energy_target.unsqueeze(1), batch['x_len'])
                pitch_cond_loss = self.ce_loss(pred['pitch_cond'].transpose(1, 2), batch['pitch_cond'])

                mel_start = batch['mel_start']
                mel_end = batch['mel_end']
                mel_batch = torch.zeros((mel_start.size(0), 1024, 64)).to(device)
                for b in range(mel_start.size(0)):
                    mel_batch[b, :, :] = pred['mel_post'][b, :, mel_start[b]:mel_end[b]]

                a, b = g_model(mel_batch)
                wav_fake = torch_stft.inverse(a, b)
                wav_real = batch['wav'].unsqueeze(1)

                d_loss = 0.0
                g_loss = 0.0
                d_fake = d_model(wav_fake.detach())
                d_real = d_model(wav_real)
                for (_, score_fake), (_, score_real) in zip(d_fake, d_real):
                    d_loss += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
                    d_loss += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()

                # generator
                d_fake = d_model(wav_fake)
                for (feat_fake, score_fake), (feat_real, _) in zip(d_fake, d_real):
                    g_loss += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                    for feat_fake_i, feat_real_i in zip(feat_fake, feat_real):
                        g_loss += 10. * F.l1_loss(feat_fake_i, feat_real_i.detach())

                loss = m1_loss + m2_loss \
                       + self.train_cfg['dur_loss_factor'] * dur_loss \
                       + self.train_cfg['pitch_loss_factor'] * pitch_loss \
                       + self.train_cfg['energy_loss_factor'] * energy_loss \
                       + self.train_cfg['pitch_cond_loss_factor'] * pitch_cond_loss \
                       + 0.1 * g_loss

                pitch_cond_true_pos = (torch.argmax(pred['pitch_cond'], dim=-1) == batch['pitch_cond'])
                pitch_cond_acc = pitch_cond_true_pos[batch['pitch_cond'] != 0].sum() / (batch['pitch_cond'] != 0).sum()

                optimizer.zero_grad()
                g_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               self.train_cfg['clip_grad_norm'])
                optimizer.step()
                g_optim.step()

                averages['mel_loss'].add(m1_loss.item() + m2_loss.item())
                averages['dur_loss'].add(dur_loss.item())
                step = model.get_step()
                k = step // 1000

                averages['step_duration'].add(time.time() - start)

                speed = 1. / averages['step_duration'].get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss: {averages["mel_loss"].get():#.4} ' \
                      f'| Dur Loss: {averages["dur_loss"].get():#.4} | {speed:#.2} steps/s | Step: {k}k | '

                if step % self.train_cfg['checkpoint_every'] == 0:
                    save_checkpoint(model=model, optim=optimizer, config=self.config,
                                    path=self.paths.forward_checkpoints / f'forward_step{k}k.pt',
                                    meta={'speaker_embeddings': self.speaker_embs})
                    torch.save({'model_g': g_model.state_dict(),
                                'model_d': d_model.state_dict()},
                               self.paths.forward_checkpoints / f'melgan_step{k}.k.pt')

                if step % self.train_cfg['plot_every'] == 0:
                    self.generate_plots(model, session, g_model, torch_stft)

                self.writer.add_scalar('g_loss/train', g_loss, model.get_step())
                self.writer.add_scalar('d_loss/train', d_loss, model.get_step())
                self.writer.add_scalar('Mel_Loss/train', m1_loss + m2_loss, model.get_step())
                self.writer.add_scalar('Pitch_Loss/train', pitch_loss, model.get_step())
                self.writer.add_scalar('Energy_Loss/train', energy_loss, model.get_step())
                self.writer.add_scalar('Duration_Loss/train', dur_loss, model.get_step())
                self.writer.add_scalar('Pitch_Cond_Loss/train', pitch_cond_loss, model.get_step())
                self.writer.add_scalar('Pitch_Cond_Accuracy/train', pitch_cond_acc, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            val_out = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Mel_Loss/val', val_out['mel_loss'], model.get_step())
            self.writer.add_scalar('Duration_Loss/val', val_out['dur_loss'], model.get_step())
            self.writer.add_scalar('Pitch_Loss/val', val_out['pitch_loss'], model.get_step())
            self.writer.add_scalar('Energy_Loss/val', val_out['energy_loss'], model.get_step())
            self.writer.add_scalar('Pitch_Cond_Loss/val', val_out['pitch_cond_loss'], model.get_step())
            self.writer.add_scalar('Pitch_Cond_Accuracy/val', val_out['pitch_cond_acc'], model.get_step())
            save_checkpoint(model=model, optim=optimizer, config=self.config,
                            path=self.paths.forward_checkpoints / 'latest_model.pt',
                            meta={'speaker_embeddings': self.speaker_embs})

            for avg in averages.values():
                avg.reset()
            print(' ')

    def evaluate(self, model: Union[MultiForwardTacotron, MultiFastPitch], val_set: DataLoader) -> Dict[str, float]:
        model.eval()
        val_losses = {
            'mel_loss': 0, 'dur_loss': 0, 'pitch_loss': 0,
            'energy_loss': 0, 'pitch_cond_loss': 0, 'pitch_cond_acc': 0
        }
        device = next(model.parameters()).device
        for i, batch in enumerate(val_set, 1):
            batch = to_device(batch, device=device)
            with torch.no_grad():
                pred = model(batch)
                m1_loss = self.l1_loss(pred['mel'], batch['mel'], batch['mel_len'])
                m2_loss = self.l1_loss(pred['mel_post'], batch['mel'], batch['mel_len'])
                dur_loss = self.l1_loss(pred['dur'].unsqueeze(1), batch['dur'].unsqueeze(1), batch['x_len'])
                pitch_loss = self.l1_loss(pred['pitch'], batch['pitch'].unsqueeze(1), batch['x_len'])
                energy_loss = self.l1_loss(pred['energy'], batch['energy'].unsqueeze(1), batch['x_len'])
                pitch_cond_loss = self.ce_loss(pred['pitch_cond'].transpose(1, 2), batch['pitch_cond'])
                pitch_cond_true_pos = (torch.argmax(pred['pitch_cond'], dim=-1) == batch['pitch_cond'])
                pitch_cond_acc = pitch_cond_true_pos[batch['pitch_cond'] != 0].sum() / (batch['pitch_cond'] != 0).sum()
                val_losses['pitch_loss'] += pitch_loss
                val_losses['energy_loss'] += energy_loss
                val_losses['mel_loss'] += m1_loss.item() + m2_loss.item()
                val_losses['dur_loss'] += dur_loss
                val_losses['pitch_cond_loss'] += pitch_cond_loss
                val_losses['pitch_cond_acc'] += pitch_cond_acc
        val_losses = {k: v / len(val_set) for k, v in val_losses.items()}
        return val_losses

    @ignore_exception
    def generate_plots(self, model: Union[MultiForwardTacotron, MultiFastPitch], session: TTSSession, model_g, torch_stft) -> None:
        model.eval()
        device = next(model.parameters()).device
        batch = session.val_sample
        batch = to_device(batch, device=device)

        pred = model(batch)
        m1_hat = np_now(pred['mel'])[0, :, :]
        m2_hat = np_now(pred['mel_post'])[0, :, :]
        m_target = np_now(batch['mel'])[0, :, :]
        speaker = batch['speaker_name'][0]


        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)
        m_target_fig = plot_mel(m_target)
        pitch_fig = plot_pitch(np_now(batch['pitch'][0]))
        pitch_gta_fig = plot_pitch(np_now(pred['pitch'].squeeze()[0]))
        energy_fig = plot_pitch(np_now(batch['energy'][0]))
        energy_gta_fig = plot_pitch(np_now(pred['energy'].squeeze()[0]))

        self.writer.add_figure(f'Pitch/target/{speaker}', pitch_fig, model.step)
        self.writer.add_figure(f'Pitch/ground_truth_aligned/{speaker}', pitch_gta_fig, model.step)
        self.writer.add_figure(f'Energy/target/{speaker}', energy_fig, model.step)
        self.writer.add_figure(f'Energy/ground_truth_aligned/{speaker}', energy_gta_fig, model.step)
        self.writer.add_figure(f'Ground_Truth_Aligned/target/{speaker}', m_target_fig, model.step)
        self.writer.add_figure(f'Ground_Truth_Aligned/linear/{speaker}', m1_hat_fig, model.step)
        self.writer.add_figure(f'Ground_Truth_Aligned/postnet/{speaker}', m2_hat_fig, model.step)

        m2_hat_wav = self.dsp.griffinlim(m2_hat)
        target_wav = self.dsp.griffinlim(m_target)

        self.writer.add_audio(
            tag=f'Ground_Truth_Aligned/target_wav/{speaker}', snd_tensor=target_wav,
            global_step=model.step, sample_rate=self.dsp.sample_rate)
        self.writer.add_audio(
            tag=f'Ground_Truth_Aligned/postnet_wav/{speaker}', snd_tensor=m2_hat_wav,
            global_step=model.step, sample_rate=self.dsp.sample_rate)

        self.writer.add_figure(f'Generated/target/{speaker}', m_target_fig, model.step)
        speakers_to_plot = self.train_cfg['plot_speakers'] + self.speakers[:self.train_cfg['plot_n_speakers']]
        speakers_to_plot = [speaker] + sorted(list({s for s in speakers_to_plot if s in self.speakers}))

        self.writer.add_audio(
            tag=f'Generated/target_wav/{speaker}', snd_tensor=target_wav,
            global_step=model.step, sample_rate=self.dsp.sample_rate)

        for speaker in speakers_to_plot:
            speaker_emb = self.speaker_embs[speaker].to(device)
            gen = model.generate(batch['x'][0:1, :batch['x_len'][0]], speaker_emb=speaker_emb)
            m2_hat = np_now(gen['mel_post'].squeeze())

            a, b = model_g.inference(gen['mel_post'])
            wav_hat = torch_stft.inverse(a, b)

            m2_hat_fig = plot_mel(m2_hat)

            pitch_gen_fig = plot_pitch(np_now(gen['pitch'].squeeze()))
            energy_gen_fig = plot_pitch(np_now(gen['energy'].squeeze()))

            self.writer.add_figure(f'Pitch/generated/{speaker}', pitch_gen_fig, model.step)
            self.writer.add_figure(f'Energy/generated/{speaker}', energy_gen_fig, model.step)
            self.writer.add_figure(f'Generated/postnet/{speaker}', m2_hat_fig, model.step)
            self.writer.add_audio(f'Generated/wav/{speaker}', wav_hat, sample_rate=22050, global_step=model.step)

            m2_hat_wav = self.dsp.griffinlim(m2_hat)

            self.writer.add_audio(
                tag=f'Generated/postnet_wav/{speaker}', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=self.dsp.sample_rate)
