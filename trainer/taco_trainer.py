import time

import torch
import torch.nn.functional as F
import tqdm
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Any

from models.aligner import Aligner
from models.tacotron import Tacotron
from trainer.common import Averager, TTSSession, to_device, np_now
from utils.checkpoints import save_checkpoint
from utils.dataset import get_taco_dataloaders
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel, plot_attention
from utils.dsp import DSP
from utils.files import parse_schedule
from utils.metrics import attention_score
from utils.paths import Paths



class ForwardSumLossOrig(torch.nn.Module):

    def __init__(self, blank_logprob=-1):
        super(ForwardSumLossOrig, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = torch.nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, text_lens, mel_lens):
        """
        Args:
        attn_logprob: batch x 1 x max(mel_lens) x max(text_lens)
        batched tensor of attention log
        probabilities, padded to length
        of longest sequence in each dimension
        text_lens: batch-D vector of length of
        each text sequence
        mel_lens: batch-D vector of length of
        each mel sequence
        """
        # The CTC loss module assumes the existence of a blank token
        # that can be optionally inserted anywhere in the sequence for
        # a fixed probability.
        # A row must be added to the attention matrix to account for this
        attn_logprob = attn_logprob.unsqueeze(1)
        attn_logprob_pd = F.pad(input=attn_logprob,
                                pad=(1, 0, 0, 0, 0, 0, 0, 0),
                                value=self.blank_logprob)
        cost_total = 0.0
        # for-loop over batch because of variable-length
        # sequences
        for bid in range(attn_logprob.shape[0]):
            # construct the target sequence. Every
            # text token is mapped to a unique sequence number,
            # thereby ensuring the monotonicity constraint
            target_seq = torch.arange(1, text_lens[bid]+1)
            target_seq=target_seq.unsqueeze(0)
            curr_logprob = attn_logprob_pd[bid].permute(1, 0, 2)
            curr_logprob = curr_logprob[:mel_lens[bid],:,:text_lens[bid]+1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            #print(attn_logprob.size(), curr_logprob.size(), target_seq.size(), mel_lens[bid:bid+1], text_lens[bid:bid+1])

            cost = self.CTCLoss(curr_logprob,
                                target_seq,
                                input_lengths=mel_lens[bid:bid+1],
                                target_lengths=text_lens[bid:bid+1])
            cost_total += cost
            # average cost over batch
        cost_total = cost_total/attn_logprob.shape[0]
        return cost_total



class ForwardSumLoss(torch.nn.Module):

    def __init__(self, blank_logprob=-1):
        super(ForwardSumLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = torch.nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, text_lens, mel_lens):
        """
        Args:
        attn_logprob: batch x 1 x max(mel_lens) x max(text_lens)
        batched tensor of attention log
        probabilities, padded to length
        of longest sequence in each dimension
        text_lens: batch-D vector of length of
        each text sequence
        mel_lens: batch-D vector of length of
        each mel sequence
        """
        # The CTC loss module assumes the existence of a blank token
        # that can be optionally inserted anywhere in the sequence for
        # a fixed probability.
        # A row must be added to the attention matrix to account for this
        attn_logprob_pd = F.pad(input=attn_logprob,
                                pad=(1, 0, 0, 0, 0, 0),
                                value=self.blank_logprob)


        bs = attn_logprob.size(0)
        T = attn_logprob.size(-1)
        target_seq = torch.arange(1, T+1).expand(bs, T)
        attn_logprob_pd = attn_logprob_pd.permute(1, 0, 2)
        attn_logprob_pd = attn_logprob_pd.log_softmax(-1)

        cost = self.CTCLoss(attn_logprob_pd,
                                target_seq,
                                input_lengths=mel_lens,
                                target_lengths=text_lens)
        return cost


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
        self.loss_fn = ForwardSumLoss()
        self.loss_fn_orig = ForwardSumLossOrig()

    def train(self,
              model: Aligner,
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
        self.loss_fn_orig = self.loss_fn_orig.to(device)
        self.loss_fn = self.loss_fn.to(device)
        for e in range(1, epochs + 1):
            for i, batch in enumerate(session.train_set, 1):
                batch = to_device(batch, device=device)
                start = time.time()
                model.train()
                attn = model(batch['x'].detach(), batch['mel'].detach(), batch['speaker_emb'])

                #loss = self.loss_fn_orig(attn, text_lens=batch['x_len'], mel_lens=batch['mel_len'])
                loss = self.loss_fn(attn, text_lens=batch['x_len'], mel_lens=batch['mel_len'])

                optimizer.zero_grad()
                if not torch.isnan(loss) or torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   self.train_cfg['clip_grad_norm'])
                    optimizer.step()

                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {loss.item():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                if step % self.train_cfg['checkpoint_every'] == 0:
                    save_checkpoint(model=model, optim=optimizer, config=self.config,
                                    path=self.paths.taco_checkpoints / f'taco_step{k}k.pt')

                if step % self.train_cfg['plot_every'] == 0:
                    self.generate_plots(model, session)

                _, att_score = attention_score(attn.softmax(-1), batch['mel_len'])
                att_score = torch.mean(att_score)
                self.writer.add_scalar('Attention_Score/train', att_score, model.get_step())
                self.writer.add_scalar('Loss/train', loss, model.get_step())
                self.writer.add_scalar('Params/reduction_factor', session.r, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            save_checkpoint(model=model, optim=optimizer, config=self.config,
                            path=self.paths.taco_checkpoints / 'latest_model.pt')

            loss_avg.reset()
            duration_avg.reset()
            print(' ')

    @ignore_exception
    def generate_plots(self, model: Tacotron, session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        batch = session.val_sample
        batch = to_device(batch, device=device)
        with torch.no_grad():
            att = model(batch['x'], batch['mel'], batch['speaker_emb']).softmax(-1)
        att = np_now(att)[0]

        att_fig = plot_attention(att)
        self.writer.add_figure('Ground_Truth_Aligned/attention', att_fig, model.step)

        """
        att_avg = torch.zeros((32, att.shape[0], att.shape[1]))
        
        model.train()
        for b in tqdm.tqdm(range(32), total=32):
            with torch.no_grad():
                att = model(batch['x'], batch['mel'])
            att_avg[b, :, :] = att[0]

        att_avg = torch.mean(att_avg, dim=0).softmax(-1)
        att_avg = np_now(att_avg)
        
        att_fig = plot_attention(att_avg)
        
        self.writer.add_figure('Ground_Truth_Aligned/attention_avg', att_fig, model.step)
        """