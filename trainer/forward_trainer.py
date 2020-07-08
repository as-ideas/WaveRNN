import time
import torch.nn.functional as F

from typing import Tuple

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from models.forward_tacotron import ForwardTacotron
from trainer.common import Averager, TTSSession, MaskedL1
from utils import hparams as hp
from utils.checkpoints import save_checkpoint
from utils.dataset import get_tts_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel
from utils.dsp import reconstruct_waveform, np_now
from utils.paths import Paths


class ForwardTrainer:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=paths.forward_log, comment='v1')
        self.l1_loss = MaskedL1()

    def train(self, model: ForwardTacotron, optimizer: Optimizer) -> None:
        for i, session_params in enumerate(hp.forward_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = get_tts_datasets(
                    path=self.paths.data, batch_size=bs, r=1, model_type='forward')
                session = TTSSession(
                    index=i, r=1, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self, model: ForwardTacotron,
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
        device = next(model.parameters()).device  # use same device as model parameters
        for e in range(1, epochs + 1):
            for i, (x, m, ids, lens, dur) in enumerate(session.train_set, 1):

                start = time.time()
                model.train()
                x, m, dur, lens = x.to(device), m.to(device), dur.to(device), lens.to(device)

                m1_hat, m2_hat, dur_hat = model(x, m, dur)

                m1_loss = self.l1_loss(m1_hat, m, lens)
                m2_loss = self.l1_loss(m2_hat, m, lens)

                dur_loss = F.l1_loss(dur_hat, dur)

                loss = m1_loss + m2_loss + dur_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                optimizer.step()
                m_loss_avg.add(m1_loss.item() + m2_loss.item())
                dur_loss_avg.add(dur_loss.item())
                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss: {m_loss_avg.get():#.4} ' \
                      f'| Dur Loss: {dur_loss_avg.get():#.4} | {speed:#.2} steps/s | Step: {k}k | '

                if step % hp.forward_checkpoint_every == 0:
                    ckpt_name = f'forward_step{k}K'
                    save_checkpoint('forward', self.paths, model, optimizer,
                                    name=ckpt_name, is_silent=True)

                if step % hp.forward_plot_every == 0:
                    self.generate_plots(model, session)

                self.writer.add_scalar('Mel_Loss/train', m1_loss + m2_loss, model.get_step())
                self.writer.add_scalar('Duration_Loss/train', dur_loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            m_val_loss, dur_val_loss = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Mel_Loss/val', m_val_loss, model.get_step())
            self.writer.add_scalar('Duration_Loss/val', dur_val_loss, model.get_step())
            save_checkpoint('forward', self.paths, model, optimizer, is_silent=True)

            m_loss_avg.reset()
            duration_avg.reset()
            print(' ')

    def evaluate(self, model: ForwardTacotron, val_set: Dataset) -> Tuple[float, float]:
        model.eval()
        m_val_loss = 0
        dur_val_loss = 0
        device = next(model.parameters()).device
        for i, (x, m, ids, lens, dur) in enumerate(val_set, 1):
            x, m, dur, lens = x.to(device), m.to(device), dur.to(device), lens.to(device)
            with torch.no_grad():
                m1_hat, m2_hat, dur_hat = model(x, m, dur)
                m1_loss = self.l1_loss(m1_hat, m, lens)
                m2_loss = self.l1_loss(m2_hat, m, lens)
                dur_loss = F.l1_loss(dur_hat, dur)
                m_val_loss += m1_loss.item() + m2_loss.item()
                dur_val_loss += dur_loss.item()
        return m_val_loss / len(val_set), dur_val_loss / len(val_set)

    @ignore_exception
    def generate_plots(self, model: ForwardTacotron, session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        x, m, ids, lens, dur = session.val_sample
        x, m, dur = x.to(device), m.to(device), dur.to(device)

        m1_hat, m2_hat, dur_hat = model(x, m, dur)
        m1_hat = np_now(m1_hat)[0, :600, :]
        m2_hat = np_now(m2_hat)[0, :600, :]
        m = np_now(m)[0, :600, :]

        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)
        m_fig = plot_mel(m)

        self.writer.add_figure('Ground_Truth_Aligned/target', m_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/linear', m1_hat_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/postnet', m2_hat_fig, model.step)

        m2_hat_wav = reconstruct_waveform(m2_hat)
        target_wav = reconstruct_waveform(m)

        self.writer.add_audio(
            tag='Ground_Truth_Aligned/target_wav', snd_tensor=target_wav,
            global_step=model.step, sample_rate=hp.sample_rate)
        self.writer.add_audio(
            tag='Ground_Truth_Aligned/postnet_wav', snd_tensor=m2_hat_wav,
            global_step=model.step, sample_rate=hp.sample_rate)

        m1_hat, m2_hat, dur_hat = model.generate(x[0].tolist())
        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)

        self.writer.add_figure('Generated/target', m_fig, model.step)
        self.writer.add_figure('Generated/linear', m1_hat_fig, model.step)
        self.writer.add_figure('Generated/postnet', m2_hat_fig, model.step)

        m2_hat_wav = reconstruct_waveform(m2_hat)

        self.writer.add_audio(
            tag='Generated/target_wav', snd_tensor=target_wav,
            global_step=model.step, sample_rate=hp.sample_rate)
        self.writer.add_audio(
            tag='Generated/postnet_wav', snd_tensor=m2_hat_wav,
            global_step=model.step, sample_rate=hp.sample_rate)
