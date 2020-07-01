import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from models.fatchord_version import WaveRNN
from trainer.common import Averager, VocSession
from utils import hparams as hp
from utils.checkpoints import save_checkpoint
from utils.dataset import get_vocoder_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table
from utils.distribution import discretized_mix_logistic_loss
from utils.dsp import decode_mu_law, label_2_float, raw_melspec
from utils.files import unpickle_binary, pickle_binary, get_files
from utils.paths import Paths


class VocTrainer:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=paths.voc_log, comment='v1')
        self.loss_func = F.cross_entropy if hp.voc_mode == 'RAW' else discretized_mix_logistic_loss
        path_top_k = paths.voc_top_k/'top_k.pkl'
        if os.path.exists(path_top_k):
            self.top_k_models = unpickle_binary(path_top_k)
            # log recent top models
            for i, (mel_loss, g_wav, m_step, m_name) in enumerate(self.top_k_models, 1):
                self.writer.add_audio(
                    tag=f'Top_K_Models/generated_top_{i}',
                    snd_tensor=g_wav, global_step=m_step, sample_rate=hp.sample_rate)
        else:
            self.top_k_models = []

    def train(self, model: WaveRNN, optimizer: Optimizer, train_gta=False) -> None:
        for i, session_params in enumerate(hp.voc_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set, val_set_samples = get_vocoder_datasets(
                    path=self.paths.data, batch_size=bs, train_gta=train_gta)
                session = VocSession(
                    index=i, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set,
                    val_set_samples=val_set_samples)
                self.train_session(model, optimizer, session, train_gta)

    def train_session(self, model: WaveRNN, optimizer: Optimizer,
                      session: VocSession, train_gta: bool) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        simple_table([(f'Steps ', str(training_steps // 1000) + 'k'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr),
                      ('Sequence Length', hp.voc_seq_len),
                      ('GTA Training', train_gta)])
        for g in optimizer.param_groups:
            g['lr'] = session.lr

        loss_avg = Averager()
        duration_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters

        for e in range(1, epochs + 1):
            for i, (x, y, m) in enumerate(session.train_set, 1):
                start = time.time()
                model.train()
                x, m, y = x.to(device), m.to(device), y.to(device)

                y_hat = model(x, m)
                if model.mode == 'RAW':
                    y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                elif model.mode == 'MOL':
                    y = y.float()
                y = y.unsqueeze(-1)

                loss = self.loss_func(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.voc_clip_grad_norm)
                optimizer.step()
                loss_avg.add(loss.item())
                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {loss_avg.get():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                if step % hp.voc_gen_samples_every == 0:
                    stream(msg + 'generating samples...')
                    gen_result = self.generate_samples(model, session)
                    if gen_result is not None:
                        mel_loss, gen_wav = gen_result
                        self.writer.add_scalar('Loss/generated_mel_l1', mel_loss, model.get_step())
                        self.track_top_models(mel_loss, gen_wav, model)

                if step % hp.voc_checkpoint_every == 0:
                    ckpt_name = f'wave_step{k}K'
                    save_checkpoint('voc', self.paths, model, optimizer,
                                    name=ckpt_name, is_silent=True)

                self.writer.add_scalar('Loss/train', loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            val_loss = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Loss/val', val_loss, model.get_step())
            save_checkpoint('voc', self.paths, model, optimizer, is_silent=True)

            loss_avg.reset()
            duration_avg.reset()
            print(' ')

    def evaluate(self, model: WaveRNN, val_set: Dataset) -> float:
        model.eval()
        val_loss = 0
        device = next(model.parameters()).device
        for i, (x, y, m) in enumerate(val_set, 1):
            x, m, y = x.to(device), m.to(device), y.to(device)
            with torch.no_grad():
                y_hat = model(x, m)
                if model.mode == 'RAW':
                    y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                elif model.mode == 'MOL':
                    y = y.float()
                y = y.unsqueeze(-1)
                loss = self.loss_func(y_hat, y)
                val_loss += loss.item()
        return val_loss / len(val_set)

    @ignore_exception
    def generate_samples(self, model: WaveRNN, session: VocSession) -> Tuple[float, list]:
        """
        Generates audio samples to cherry-pick models. To evaluate audio quality
        we calculate the l1 distance between mels of predictions and targets.
        """
        model.eval()
        mel_losses = []
        gen_wavs = []
        device = next(model.parameters()).device
        for i, (m, x) in enumerate(session.val_set_samples, 1):
            if i > hp.voc_gen_num_samples:
                break
            x = x[0].numpy()
            bits = 16 if hp.voc_mode == 'MOL' else hp.bits
            if hp.mu_law and hp.voc_mode != 'MOL':
                x = decode_mu_law(x, 2 ** bits, from_labels=True)
            else:
                x = label_2_float(x, bits)
            gen_wav = model.generate(
                mels=m, save_path=None, batched=hp.voc_gen_batched,
                target=hp.voc_target, overlap=hp.voc_overlap,
                mu_law=hp.mu_law, silent=True)

            gen_wavs.append(gen_wav)
            y_mel = raw_melspec(x.squeeze())
            y_mel = torch.tensor(y_mel).to(device)
            y_hat_mel = raw_melspec(gen_wav)
            y_hat_mel = torch.tensor(y_hat_mel).to(device)
            loss = F.l1_loss(y_hat_mel, y_mel)
            mel_losses.append(loss.item())

            self.writer.add_audio(
                tag=f'Validation_Samples/target_{i}', snd_tensor=x,
                global_step=model.step, sample_rate=hp.sample_rate)
            self.writer.add_audio(
                tag=f'Validation_Samples/generated_{i}',
                snd_tensor=gen_wav, global_step=model.step, sample_rate=hp.sample_rate)

        return sum(mel_losses) / len(mel_losses), gen_wavs[0]

    def track_top_models(self, mel_loss, gen_wav, model):
        """ Keeps track of top k models and saves them according to their current rank """
        print()
        print(f'mel loss {mel_loss}')
        for j, (l, g, m, m_n) in enumerate(self.top_k_models):
            print(f'{j} {l} {m} {m_n}')
        if len(self.top_k_models) < hp.voc_keep_top_k or mel_loss < self.top_k_models[-1][0]:
            m_step = model.get_step()
            model_name = f'model_loss{mel_loss:#0.5}_step{m_step}_weights.pyt'
            self.top_k_models.append((mel_loss, gen_wav, model.get_step(), model_name))
            self.top_k_models.sort(key=lambda t: t[0])
            self.top_k_models = self.top_k_models[:hp.voc_keep_top_k]
            model.save(self.paths.voc_top_k/model_name)
            all_models = get_files(self.paths.voc_top_k, extension='pyt')
            top_k_names = {m[-1] for m in self.top_k_models}
            for model_file in all_models:
                if model_file.name not in top_k_names:
                    print(f'removing {model_file}')
                    os.remove(model_file)
            pickle_binary(self.top_k_models, self.paths.voc_top_k/'top_k.pkl')

            for i, (mel_loss, g_wav, m_step, m_name) in enumerate(self.top_k_models, 1):
                self.writer.add_audio(
                    tag=f'Top_K_Models/generated_top_{i}',
                    snd_tensor=g_wav, global_step=m_step, sample_rate=hp.sample_rate)

        print('sorted ranks:')
        for j, (l, g, m, m_n) in enumerate(self.top_k_models):
            print(f'{j} {l} {m} {m_n}')
