import math
import time

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Any

from models.tacotron import Tacotron
from trainer.common import Averager, TTSSession, to_device, np_now, ForwardSumLoss, new_guided_attention_matrix
from utils.checkpoints import save_checkpoint
from utils.dataset import get_taco_dataloaders
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel, plot_attention
from utils.dsp import DSP
from utils.files import parse_schedule
from utils.metrics import attention_score
from utils.paths import Paths


class TacoTrainer:

    def __init__(self,
                 paths: Paths,
                 dsp: DSP,
                 config: Dict[str, Any]) -> None:
        self.paths = paths
        self.dsp = dsp
        self.config = config
        self.train_cfg = config['tacotron']['training']
        self.writer = SummaryWriter(log_dir=paths.taco_log, comment='v1')
        self.forward_loss = ForwardSumLoss()

    def train(self,
              model: Tacotron,
              optimizer: Optimizer) -> None:
        tts_schedule = self.train_cfg['schedule']
        tts_schedule = parse_schedule(tts_schedule)
        for i, session_params in enumerate(tts_schedule, 1):
            r, lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = get_taco_dataloaders(
                    paths=self.paths, batch_size=bs, r=r,
                    **self.train_cfg['filter']
                )
                session = TTSSession(
                    index=i, r=r, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session=session)

    def train_session(self, model: Tacotron,
                      optimizer: Optimizer,
                      session: TTSSession) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        model.r = session.r
        simple_table([(f'Steps with r={session.r}', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr),
                      ('Outputs/Step (r)', model.r)])
        for g in optimizer.param_groups:
            g['lr'] = session.lr

        loss_avg = Averager()
        duration_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters
        for e in range(1, epochs + 1):
            for i, batch in enumerate(session.train_set, 1):
                batch = to_device(batch, device=device)
                start = time.time()
                model.train()

                out = model(batch)
                m1_hat, m2_hat, attention, att_aligner = out['mel'], out['mel_post'], out['att'], out['att_aligner']
                ctc_loss = self.forward_loss(att_aligner, text_lens=batch['x_len'], mel_lens=batch['mel_len'])

                m1_loss = F.l1_loss(m1_hat, batch['mel'])
                m2_loss = F.l1_loss(m2_hat, batch['mel'])

                dia_mat = new_guided_attention_matrix(attention=attention,
                                                      g=self.train_cfg['dia_loss_matrix_g'])
                dia_loss = ((1 - dia_mat) * attention).mean()

                mel_loss = m1_loss + m2_loss
                loss = mel_loss + self.train_cfg['ctc_loss_factor'] * ctc_loss \
                       + self.train_cfg['dia_loss_factor'] * dia_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               self.train_cfg['clip_grad_norm'])
                optimizer.step()
                loss_avg.add(loss.item())
                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {loss_avg.get():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                if step % self.train_cfg['checkpoint_every'] == 0:
                    save_checkpoint(model=model, optim=optimizer, config=self.config,
                                    path=self.paths.taco_checkpoints / f'taco_step{k}k.pt')

                if step % self.train_cfg['plot_every'] == 0:
                    self.generate_plots(model, session)

                _, att_score = attention_score(attention, batch['mel_len'])
                att_score = torch.mean(att_score)
                self.writer.add_scalar('Attention_Score/train', att_score, model.get_step())
                self.writer.add_scalar('Mel_Loss/train', mel_loss, model.get_step())
                self.writer.add_scalar('CTC_Loss/train', ctc_loss, model.get_step())
                self.writer.add_scalar('Dia_Loss/train', dia_loss, model.get_step())
                self.writer.add_scalar('Params/reduction_factor', session.r, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            val_loss, val_att_score = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Mel_Loss/val', val_loss, model.get_step())
            self.writer.add_scalar('Attention_Score/val', val_att_score, model.get_step())
            save_checkpoint(model=model, optim=optimizer, config=self.config,
                            path=self.paths.taco_checkpoints / 'latest_model.pt')

            loss_avg.reset()
            duration_avg.reset()
            print(' ')

    def evaluate(self, model: Tacotron, val_set: DataLoader) -> Tuple[float, float]:
        model.eval()
        model.decoder.prenet.train()
        val_loss = 0
        val_att_score = 0
        device = next(model.parameters()).device
        for i, batch in enumerate(val_set, 1):
            batch = to_device(batch, device=device)
            with torch.no_grad():
                out = model(batch)
                m1_hat, m2_hat, attention = out['mel'], out['mel_post'], out['att']
                m1_loss = F.l1_loss(m1_hat, batch['mel'])
                m2_loss = F.l1_loss(m2_hat, batch['mel'])
                val_loss += m1_loss.item() + m2_loss.item()
            _, att_score = attention_score(attention, batch['mel_len'])
            val_att_score += torch.mean(att_score).item()

        return val_loss / len(val_set), val_att_score / len(val_set)

    @ignore_exception
    def generate_plots(self, model: Tacotron, session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        batch = session.val_sample
        batch = to_device(batch, device=device)
        with torch.no_grad():
            out = model(batch)
        m1_hat, m2_hat, att, att_aligner = out['mel'], out['mel_post'], out['att'], out['att_aligner']
        att = np_now(att)[0]
        att_aligner = np_now(att_aligner.softmax(-1))[0]
        m1_hat = np_now(m1_hat)[0, :, :]
        m2_hat = np_now(m2_hat)[0, :, :]
        m_target = np_now(batch['mel'])[0, :, :]
        speaker = batch['speaker_name'][0]

        att_fig = plot_attention(att)
        att_aligner_fig = plot_attention(att_aligner)

        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)
        m_target_fig = plot_mel(m_target)

        self.writer.add_figure(f'Ground_Truth_Aligned/attention/{speaker}', att_fig, model.step)
        self.writer.add_figure(f'Ground_Truth_Aligned/attention_aligner/{speaker}', att_aligner_fig, model.step)
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