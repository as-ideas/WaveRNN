import time
from typing import Tuple, Dict, Any, Union

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.fast_pitch import FastPitch
from models.forward_tacotron import ForwardTacotron
from trainer.common import Averager, TTSSession, MaskedL1, to_device, np_now
from utils.checkpoints import  save_checkpoint
from utils.dataset import get_tts_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel, plot_pitch
from utils.dsp import DSP
from utils.files import parse_schedule
from utils.paths import Paths


class ForwardTrainer:

    def __init__(self,
                 paths: Paths,
                 dsp: DSP,
                 config: Dict[str, Any]) -> None:
        self.paths = paths
        self.dsp = dsp
        self.config = config
        model_type = config.get('tts_model', 'forward_tacotron')
        self.train_cfg = config[model_type]['training']
        self.writer = SummaryWriter(log_dir=paths.forward_log, comment='v1')
        self.l1_loss = MaskedL1()

    def train(self, model: Union[ForwardTacotron, FastPitch], optimizer: Optimizer) -> None:
        forward_schedule = self.train_cfg['schedule']
        forward_schedule = parse_schedule(forward_schedule)
        for i, session_params in enumerate(forward_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = get_tts_datasets(
                    path=self.paths.data, batch_size=bs, r=1, model_type='forward',
                    max_mel_len=self.train_cfg['max_mel_len'],
                    filter_attention=self.train_cfg['filter_attention'],
                    filter_min_alignment=self.train_cfg['min_attention_alignment'],
                    filter_min_sharpness=self.train_cfg['min_attention_sharpness'],
                    filter_min_text_prob=self.train_cfg['min_trans_prob'],
                    filter_min_text_sim=self.train_cfg['min_trans_sim'])
                session = TTSSession(
                    index=i, r=1, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self,  model: Union[ForwardTacotron, FastPitch],
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

        m_loss_avg = Averager()
        dur_loss_avg = Averager()
        duration_avg = Averager()
        pitch_loss_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters
        for e in range(1, epochs + 1):
            for i, batch in enumerate(session.train_set, 1):
                batch = to_device(batch, device=device)
                start = time.time()
                model.train()

                pitch_target = batch['pitch_hat'].detach().clone()

                #pred = model(batch)
                dur_hat = model.dur_pred(batch['x'], batch['speaker_emb'])
                pitch_hat = model.pitch_pred(batch['x'], batch['speaker_emb']).transpose(1, 2)

                dur_diff_mean = 0
                dur_diff_median = 0

                model.step += 1

                #pe = torch.sqrt((torch.abs(pitch_target.unsqueeze(1)))).detach() + 0.1

                #pe_dur = (torch.abs(dur_hat.unsqueeze(1))).detach()

                for b in range(pitch_hat.size(0)):
                    x_len = int(batch['x_len'][b])
                    #x_len = max(x_len-10, 1)
                    #pe[b, :, :x_len] = 1.
                    #pe[b, :, x_len:] += 0.1
                    dur_diff = dur_hat[b, :x_len, 0] - batch['dur_hat'][b, :x_len]
                    dur_diff_mean += torch.mean(dur_diff)
                    dur_diff_median += torch.median(dur_diff)

                dur_loss = self.l1_loss(dur_hat.transpose(1, 2), batch['dur'].unsqueeze(1), batch['x_len'])


                pitch_loss = self.l1_loss(pitch_hat*pe, pitch_target.unsqueeze(1)*pe, batch['x_len'])

                loss = self.train_cfg['pitch_loss_factor'] * pitch_loss \
                       + self.train_cfg['dur_loss_factor'] * dur_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               self.train_cfg['clip_grad_norm'])
                optimizer.step()

                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                pitch_loss_avg.add(pitch_loss.item())

                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss: {m_loss_avg.get():#.4} ' \
                      f'| Dur Loss: {dur_loss_avg.get():#.4} | Pitch Loss: {pitch_loss_avg.get():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                if step % self.train_cfg['checkpoint_every'] == 0:
                    save_checkpoint(model=model, optim=optimizer, config=self.config,
                                    path=self.paths.forward_checkpoints / f'forward_step{k}k.pt')

                if step % self.train_cfg['plot_every'] == 0:
                    self.generate_plots(model, session)

                self.writer.add_scalar('Pitch_Loss/train', pitch_loss, model.get_step())
                self.writer.add_scalar('Duration_Loss/train', dur_loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())
                self.writer.add_scalar('Dur_Diff/mean', dur_diff_mean, model.get_step())
                self.writer.add_scalar('Dur_Diff/median', dur_diff_median, model.get_step())

                stream(msg)

            val_out = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Pitch_Loss/val', val_out['pitch_loss'], model.get_step())
            self.writer.add_scalar('Duration_Loss/val', val_out['dur_loss'], model.get_step())
            save_checkpoint(model=model, optim=optimizer, config=self.config,
                            path=self.paths.forward_checkpoints / 'latest_model.pt')

            m_loss_avg.reset()
            duration_avg.reset()
            pitch_loss_avg.reset()
            print(' ')

    def evaluate(self, model: Union[ForwardTacotron, FastPitch], val_set: DataLoader) -> Dict[str, float]:
        model.eval()
        pitch_val_loss = 0
        dur_val_loss = 0
        device = next(model.parameters()).device
        for i, batch in enumerate(val_set, 1):
            batch = to_device(batch, device=device)
            with torch.no_grad():
                pitch_hat = model.pitch_pred(batch['x'], batch['speaker_emb']).transpose(1, 2)
                dur_hat = model.dur_pred(batch['x'], batch['speaker_emb'])
                pitch_loss = self.l1_loss(pitch_hat, batch['pitch_hat'].unsqueeze(1), batch['x_len'])
                dur_loss = self.l1_loss(dur_hat.transpose(1, 2), batch['dur_hat'].unsqueeze(1), batch['x_len'])
                pitch_val_loss += pitch_loss
                dur_val_loss += dur_loss
        return {
            'pitch_loss': pitch_val_loss / len(val_set),
            'dur_loss': dur_val_loss / len(val_set),
        }

    @ignore_exception
    def generate_plots(self, model: Union[ForwardTacotron, FastPitch], session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        batch = session.val_sample
        batch = to_device(batch, device=device)

        pred = model(batch)
        m1_hat = np_now(pred['mel'])[0, :600, :]
        m2_hat = np_now(pred['mel_post'])[0, :600, :]
        m_target = np_now(batch['mel'])[0, :600, :]

        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)
        m_target_fig = plot_mel(m_target)
        pitch_fig = plot_pitch(np_now(batch['pitch'][0]))
        pitch_gta_fig = plot_pitch(np_now(pred['pitch'].squeeze()[0]))
        energy_fig = plot_pitch(np_now(batch['energy'][0]))
        energy_gta_fig = plot_pitch(np_now(pred['energy'].squeeze()[0]))

        self.writer.add_figure('Pitch/target', pitch_fig, model.step)
        self.writer.add_figure('Pitch/ground_truth_aligned', pitch_gta_fig, model.step)
        self.writer.add_figure('Energy/target', energy_fig, model.step)
        self.writer.add_figure('Energy/ground_truth_aligned', energy_gta_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/target', m_target_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/linear', m1_hat_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/postnet', m2_hat_fig, model.step)

        m2_hat_wav = self.dsp.griffinlim(m2_hat)
        target_wav = self.dsp.griffinlim(m_target)

        self.writer.add_audio(
            tag='Ground_Truth_Aligned/target_wav', snd_tensor=target_wav,
            global_step=model.step, sample_rate=self.dsp.sample_rate)
        self.writer.add_audio(
            tag='Ground_Truth_Aligned/postnet_wav', snd_tensor=m2_hat_wav,
            global_step=model.step, sample_rate=self.dsp.sample_rate)

        #speaker_names = list(self.config['speaker_names'])[:10]
        speaker_names = ['welt', 'bild', 'kammmacher', '4815_komoedie3_28_dante_64kb']
        for speaker_name in speaker_names:
            speaker_emb = getattr(model, speaker_name).unsqueeze(0)
            gen = model.generate(batch['x'][0:1, :batch['x_len'][0]], semb=speaker_emb)
            m1_hat = np_now(gen['mel'].squeeze())
            m2_hat = np_now(gen['mel_post'].squeeze())

            m1_hat_fig = plot_mel(m1_hat)
            m2_hat_fig = plot_mel(m2_hat)

            pitch_gen_fig = plot_pitch(np_now(gen['pitch'].squeeze()))
            energy_gen_fig = plot_pitch(np_now(gen['energy'].squeeze()))

            self.writer.add_figure(f'{speaker_name}_pitch/generated', pitch_gen_fig, model.step)
            self.writer.add_figure(f'{speaker_name}_energy/generated', energy_gen_fig, model.step)
            self.writer.add_figure(f'{speaker_name}_generated/target', m_target_fig, model.step)
            self.writer.add_figure(f'{speaker_name}_generated/linear', m1_hat_fig, model.step)
            self.writer.add_figure(f'{speaker_name}_generated/postnet', m2_hat_fig, model.step)

            m2_hat_wav = self.dsp.griffinlim(m2_hat)

            self.writer.add_audio(
                tag=f'{speaker_name}_generated/target_wav', snd_tensor=target_wav,
                global_step=model.step, sample_rate=self.dsp.sample_rate)
            self.writer.add_audio(
                tag=f'{speaker_name}_generated/postnet_wav', snd_tensor=m2_hat_wav,
                global_step=model.step, sample_rate=self.dsp.sample_rate)