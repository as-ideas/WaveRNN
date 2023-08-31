import time
from typing import Dict, Any, Union
import torch.nn.functional as F
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.multi_fast_pitch import MultiFastPitch
from models.multi_forward_tacotron import MultiForwardTacotron, Discriminator
from trainer.common import Averager, TTSSession, MaskedL1, to_device, np_now
from utils.checkpoints import save_checkpoint
from utils.dataset import get_forward_dataloaders
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel, plot_pitch
from utils.dsp import DSP
from utils.files import parse_schedule, unpickle_binary
from utils.paths import Paths


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

    def train(self, model: Union[MultiForwardTacotron, MultiFastPitch], optimizer: Optimizer) -> None:
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

        disc = Discriminator().to(device)
        d_optim = Adam(disc.parameters(), lr=5e-5)
        mod_optim = Adam(
            list(model.dur_pred.parameters())
            + list(model.pitch_pred.parameters())
            + list(model.energy_pred.parameters())
            + list(model.pitch_cond_pred.parameters()),
            lr=5e-5)

        for e in range(1, epochs + 1):
            for i, batch in enumerate(session.train_set, 1):
                batch = to_device(batch, device=device)
                start = time.time()
                model.train()

                pitch_target = batch['pitch'].detach().clone()
                energy_target = batch['energy'].detach().clone()


                pitch_cond_hat = model.pitch_cond_pred(batch['x'], batch['speaker_emb']).squeeze(-1)
                dur_hat = model.dur_pred(batch['x'], batch['pitch_cond'], batch['speaker_emb']).squeeze(-1)
                pitch_hat = model.pitch_pred(batch['x'], batch['pitch_cond'], batch['speaker_emb']).transpose(1, 2)
                energy_hat = model.energy_pred(batch['x'], batch['speaker_emb']).transpose(1, 2)

                d_loss = 0
                score_fake = disc(batch['x'], dur_hat.unsqueeze(-1).detach(),
                                  pitch_hat.transpose(1, 2).detach(), batch['speaker_emb'], batch['pitch_cond'])
                score_real = disc(batch['x'], batch['dur'].unsqueeze(-1),
                                  batch['pitch'].unsqueeze(-1), batch['speaker_emb'], batch['pitch_cond'])

                for b in range(batch['x'].size(0)):
                    d_loss += torch.pow(score_real[b, :batch['x_len'][b], :] - 1.0, 2).mean()
                    d_loss += torch.pow(score_fake[b, :batch['x_len'][b], :], 2).mean()

                d_loss /= batch['x'].size(0)
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()

                score_fake = disc(batch['x'], dur_hat.unsqueeze(-1), pitch_hat.transpose(1, 2),
                                  batch['speaker_emb'], batch['pitch_cond'])
                g_loss = 0
                for b in range(batch['x'].size(0)):
                    g_loss += torch.pow(score_fake[b, :batch['x_len'][b], :] - 1.0, 2).mean()
                g_loss /= batch['x'].size(0)

                #print('g_loss', g_loss)
                #print('d_loss', d_loss)

                dur_loss = self.l1_loss(dur_hat.unsqueeze(1), batch['dur'].unsqueeze(1), batch['x_len'])
                pitch_loss = self.l1_loss(pitch_hat, pitch_target.unsqueeze(1), batch['x_len'])
                energy_loss = self.l1_loss(energy_hat, energy_target.unsqueeze(1), batch['x_len'])
                pitch_cond_loss = self.ce_loss(pitch_cond_hat.transpose(1, 2), batch['pitch_cond'])

                loss_modules = dur_loss + pitch_loss + energy_loss + pitch_cond_loss + g_loss
                mod_optim.zero_grad()
                loss_modules.backward()
                mod_optim.step()

                pred = model(batch)
                m1_loss = self.l1_loss(pred['mel'], batch['mel'], batch['mel_len'])
                m2_loss = self.l1_loss(pred['mel_post'], batch['mel'], batch['mel_len'])

                loss = m1_loss + m2_loss
                pitch_cond_true_pos = (torch.argmax(pitch_cond_hat, dim=-1) == batch['pitch_cond'])
                pitch_cond_acc = pitch_cond_true_pos[batch['pitch_cond'] != 0].sum() / (batch['pitch_cond'] != 0).sum()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               self.train_cfg['clip_grad_norm'])
                optimizer.step()

                averages['mel_loss'].add(m1_loss.item() + m2_loss.item())
                averages['dur_loss'].add(dur_loss.item())
                step = model.get_step()
                k = step // 1000

                averages['step_duration'].add(time.time() - start)

                speed = 1. / averages['step_duration'].get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss: {averages["mel_loss"].get():#.4} ' \
                      f'| Dur Loss: {averages["dur_loss"].get():#.4} | {speed:#.2} steps/s | Step: {model.get_step()} | '

                if step % self.train_cfg['checkpoint_every'] == 0:
                    save_checkpoint(model=model, optim=optimizer, config=self.config,
                                    path=self.paths.forward_checkpoints / f'forward_step{k}k.pt',
                                    meta={'speaker_embeddings': self.speaker_embs})

                if step % self.train_cfg['plot_every'] == 0:
                    print('generate plots...')
                    self.generate_plots(model, session)

                self.writer.add_scalar('d_loss/train', d_loss, model.get_step())
                self.writer.add_scalar('g_loss/train', g_loss, model.get_step())
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

                pitch_cond_hat = model.pitch_cond_pred(batch['x'], batch['speaker_emb']).squeeze(-1)
                dur_hat = model.dur_pred(batch['x'], batch['pitch_cond'], batch['speaker_emb']).squeeze(-1)
                pitch_hat = model.pitch_pred(batch['x'], batch['pitch_cond'], batch['speaker_emb']).transpose(1, 2)
                energy_hat = model.energy_pred(batch['x'], batch['speaker_emb']).transpose(1, 2)

                m1_loss = self.l1_loss(pred['mel'], batch['mel'], batch['mel_len'])
                m2_loss = self.l1_loss(pred['mel_post'], batch['mel'], batch['mel_len'])
                dur_loss = self.l1_loss(dur_hat.unsqueeze(1), batch['dur'].unsqueeze(1), batch['x_len'])
                pitch_loss = self.l1_loss(pitch_hat, batch['pitch'].unsqueeze(1), batch['x_len'])
                energy_loss = self.l1_loss(energy_hat, batch['energy'].unsqueeze(1), batch['x_len'])
                pitch_cond_loss = self.ce_loss(pitch_cond_hat.transpose(1, 2), batch['pitch_cond'])
                pitch_cond_true_pos = (torch.argmax(pitch_cond_hat, dim=-1) == batch['pitch_cond'])
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
    def generate_plots(self, model: Union[MultiForwardTacotron, MultiFastPitch], session: TTSSession) -> None:
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
        energy_fig = plot_pitch(np_now(batch['energy'][0]))

        self.writer.add_figure(f'Pitch/target/{speaker}', pitch_fig, model.step)
        self.writer.add_figure(f'Energy/target/{speaker}', energy_fig, model.step)
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

            m2_hat_fig = plot_mel(m2_hat)

            pitch_gen_fig = plot_pitch(np_now(gen['pitch'].squeeze()))
            energy_gen_fig = plot_pitch(np_now(gen['energy'].squeeze()))

            self.writer.add_figure(f'Pitch/generated/{speaker}', pitch_gen_fig, model.step)
            self.writer.add_figure(f'Energy/generated/{speaker}', energy_gen_fig, model.step)
            self.writer.add_figure(f'Generated/postnet/{speaker}', m2_hat_fig, model.step)

            m2_hat_wav = self.dsp.griffinlim(m2_hat)

            self.writer.add_audio(
                tag=f'Generated/postnet_wav/{speaker}', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=self.dsp.sample_rate)
