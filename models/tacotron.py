from pathlib import Path
from typing import Union, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding

from models.common_layers import CBHG
from utils.text.symbols import phonemes


class Aligner(nn.Module):

    def __init__(self,
                 num_chars: int,
                 speaker_emb_dim: int,
                 mel_dim: int,
                 hidden_dim: int,
                 out_dim: int):

        """
        Args:
            num_chars: Number of characters for embedding.
            speaker_emb_dim: Dimension of speaker embedding.
            mel_dim: Mel dimensions.
            hidden_dim: Hidden dimensions of aligner model.
            out_dim: Out dimensions of aligner model.
        """

        super().__init__()
        self.embedding = Embedding(num_embeddings=num_chars, embedding_dim=hidden_dim)
        self.text_encoder = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim + speaker_emb_dim,
                      out_channels=hidden_dim,
                      kernel_size=3,
                      padding=1))
        self.mel_encoder = nn.Sequential(
            nn.Conv1d(in_channels=mel_dim, out_channels=hidden_dim,
                      kernel_size=3, padding=1),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim,
                      kernel_size=3, padding=1),)
        self.text_lin = nn.Linear(hidden_dim, out_dim)
        self.mel_lin = nn.Linear(hidden_dim, out_dim)
        self.speaker_emb_dim = speaker_emb_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch['x']
        speaker_emb = batch['speaker_emb']
        mel = batch['mel']

        x = self.embedding(x).transpose(1, 2)
        if self.speaker_emb_dim > 0:
            speaker_emb = speaker_emb[:, :, None]
            speaker_emb = speaker_emb.repeat(1, 1, x.shape[2])
            x = torch.cat([x, speaker_emb], dim=1)

        x = self.text_encoder(x).transpose(1, 2)
        mel = self.mel_encoder(mel).transpose(1, 2)
        x = self.text_lin(x)
        mel = self.mel_lin(mel)

        diff = x[:, None, :, :] - mel[:, :, None, :]
        att = -torch.linalg.norm(diff, ord=2, dim=-1)

        return {'x': x, 'mel': mel, 'att': att}


class Encoder(nn.Module):
    def __init__(self, embed_dims, num_chars, cbhg_channels, K, num_highways, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dims)
        self.pre_net = PreNet(embed_dims)
        self.cbhg = CBHG(K=K, in_channels=cbhg_channels, channels=cbhg_channels,
                         proj_channels=[cbhg_channels, cbhg_channels],
                         num_highways=num_highways)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pre_net(x)
        x.transpose_(1, 2)
        x = self.cbhg(x)
        return x


class PreNet(nn.Module):
    def __init__(self, in_dims, fc1_dims=256, fc2_dims=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.p = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        return x


class Attention(nn.Module):
    def __init__(self, attn_dims):
        super().__init__()
        self.W = nn.Linear(attn_dims, attn_dims, bias=False)
        self.v = nn.Linear(attn_dims, 1, bias=False)

    def forward(self, encoder_seq_proj, query, t):

        # print(encoder_seq_proj.shape)
        # Transform the query vector
        query_proj = self.W(query).unsqueeze(1)

        # Compute the scores
        u = self.v(torch.tanh(encoder_seq_proj + query_proj))
        scores = F.softmax(u, dim=1)

        return scores.transpose(1, 2)


class LSA(nn.Module):
    def __init__(self, attn_dim, kernel_size=31, filters=32):
        super().__init__()
        self.conv = nn.Conv1d(2, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=False)
        self.L = nn.Linear(filters, attn_dim, bias=True)
        self.W = nn.Linear(attn_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.cumulative = None
        self.attention = None

    def init_attention(self, encoder_seq_proj):
        device = next(self.parameters()).device  # use same device as parameters
        b, t, c = encoder_seq_proj.size()
        self.cumulative = torch.zeros(b, t, device=device)
        self.attention = torch.zeros(b, t, device=device)

    def forward(self, encoder_seq_proj, query, t):

        if t == 0: self.init_attention(encoder_seq_proj)

        processed_query = self.W(query).unsqueeze(1)

        location = torch.cat([self.cumulative.unsqueeze(1), self.attention.unsqueeze(1)], dim=1)
        processed_loc = self.L(self.conv(location).transpose(1, 2))

        u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc))
        u = u.squeeze(-1)

        # Smooth Attention
        #scores = torch.sigmoid(u) / torch.sigmoid(u).sum(dim=1, keepdim=True)
        scores = F.softmax(u, dim=1)
        self.attention = scores
        self.cumulative += self.attention

        return scores.unsqueeze(-1).transpose(1, 2)


class Decoder(nn.Module):
    # Class variable because its value doesn't change between classes
    # yet ought to be scoped by class because its a property of a Decoder
    max_r = 20

    def __init__(self,
                 n_mels: int,
                 decoder_dims: int,
                 lstm_dims: int,
                 aligner_out_dims: int):
        super().__init__()
        self.register_buffer('r', torch.tensor(1, dtype=torch.int))
        self.n_mels = n_mels
        self.prenet = PreNet(n_mels + aligner_out_dims)
        self.attn_net = LSA(decoder_dims)
        self.attn_rnn = nn.GRUCell(decoder_dims + decoder_dims // 2, decoder_dims)
        self.rnn_input = nn.Linear(2 * decoder_dims, lstm_dims)
        self.res_rnn1 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.res_rnn2 = nn.LSTMCell(lstm_dims, lstm_dims)
        self.mel_proj = nn.Linear(lstm_dims, n_mels * self.max_r, bias=False)
    
    def zoneout(self, prev, current, p=0.1):
        device = next(self.parameters()).device  # Use same device as parameters
        mask = torch.zeros(prev.size(), device=device).bernoulli_(p)
        return prev * mask + current * (1 - mask)

    def forward(self, encoder_seq, encoder_seq_proj, prenet_in,
                hidden_states, cell_states, context_vec, t):

        # Need this for reshaping mels
        batch_size = encoder_seq.size(0)

        # Unpack the hidden and cell states
        attn_hidden, rnn1_hidden, rnn2_hidden = hidden_states
        rnn1_cell, rnn2_cell = cell_states

        # PreNet for the Attention RNN
        prenet_out = self.prenet(prenet_in)

        # Compute the Attention RNN hidden state
        attn_rnn_in = torch.cat([context_vec, prenet_out], dim=-1)
        attn_hidden = self.attn_rnn(attn_rnn_in.squeeze(1), attn_hidden)

        # Compute the attention scores
        scores = self.attn_net(encoder_seq_proj, attn_hidden, t)

        # Dot product to create the context vector
        context_vec = scores @ encoder_seq
        context_vec = context_vec.squeeze(1)

        # Concat Attention RNN output w. Context Vector & project
        x = torch.cat([context_vec, attn_hidden], dim=1)
        x = self.rnn_input(x)

        # Compute first Residual RNN
        rnn1_hidden_next, rnn1_cell = self.res_rnn1(x, (rnn1_hidden, rnn1_cell))
        if self.training:
            rnn1_hidden = self.zoneout(rnn1_hidden, rnn1_hidden_next)
        else:
            rnn1_hidden = rnn1_hidden_next
        x = x + rnn1_hidden

        # Compute second Residual RNN
        rnn2_hidden_next, rnn2_cell = self.res_rnn2(x, (rnn2_hidden, rnn2_cell))
        if self.training:
            rnn2_hidden = self.zoneout(rnn2_hidden, rnn2_hidden_next)
        else:
            rnn2_hidden = rnn2_hidden_next
        x = x + rnn2_hidden

        # Project Mels
        mels = self.mel_proj(x)
        mels = mels.view(batch_size, self.n_mels, self.max_r)[:, :, :self.r]
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)
        cell_states = (rnn1_cell, rnn2_cell)

        return mels, scores, hidden_states, cell_states, context_vec


class Tacotron(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 num_chars: int,
                 encoder_dims: int,
                 decoder_dims: int,
                 n_mels: int,
                 postnet_dims: int,
                 encoder_k: int,
                 lstm_dims: int,
                 postnet_k: int,
                 num_highways: int,
                 dropout: float,
                 stop_threshold: float,
                 aligner_hidden_dims: int,
                 aligner_out_dims: int,
                 speaker_emb_dim=256) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.lstm_dims = lstm_dims
        self.decoder_dims = decoder_dims
        self.encoder = Encoder(embed_dims, num_chars, encoder_dims,
                               encoder_k, num_highways, dropout)
        self.encoder_proj_query = nn.Linear(decoder_dims + speaker_emb_dim + aligner_out_dims, decoder_dims,
                                            bias=False)
        self.encoder_proj = nn.Linear(decoder_dims + speaker_emb_dim + aligner_out_dims, decoder_dims,
                                      bias=False)
        self.decoder = Decoder(n_mels=n_mels, decoder_dims=decoder_dims,
                               lstm_dims=lstm_dims, aligner_out_dims=aligner_out_dims)
        self.postnet = CBHG(postnet_k, n_mels, postnet_dims, [256, 80], num_highways)
        self.post_proj = nn.Linear(postnet_dims * 2, n_mels, bias=False)
        self.speaker_emb_dim = speaker_emb_dim

        self.init_model()

        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('stop_threshold', torch.tensor(stop_threshold, dtype=torch.float32))

        self.aligner = Aligner(num_chars=num_chars, mel_dim=n_mels, speaker_emb_dim=speaker_emb_dim,
                               hidden_dim=aligner_hidden_dims, out_dim=aligner_out_dims)

        self.aligner_out_dims = aligner_out_dims

    @property
    def r(self) -> int:
        return self.decoder.r.item()

    @r.setter
    def r(self, value: int) -> None:
        self.decoder.r = self.decoder.r.new_tensor(value, requires_grad=False)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.tensor:
        x = batch['x']
        mel = batch['mel']
        speaker_emb = batch['speaker_emb']
        device = next(self.parameters()).device  # use same device as parameters

        if self.training:
            self.step += 1

        batch_size, _, steps = mel.size()

        # Initialise all hidden states and pack into tuple
        attn_hidden = torch.zeros(batch_size, self.decoder_dims, device=device)
        rnn1_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_hidden = torch.zeros(batch_size, self.lstm_dims, device=device)
        hidden_states = (attn_hidden, rnn1_hidden, rnn2_hidden)

        # Initialise all lstm cell states and pack into tuple
        rnn1_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        rnn2_cell = torch.zeros(batch_size, self.lstm_dims, device=device)
        cell_states = (rnn1_cell, rnn2_cell)

        # <GO> Frame for start of decoder loop
        go_frame = torch.zeros(batch_size, self.n_mels + self.aligner_out_dims, device=device)

        # Need an initial context vector
        context_vec = torch.zeros(batch_size, self.decoder_dims, device=device)

        # get aligner output that produces encoded x and mel to be concatenated to taco inputs
        aligner_out = self.aligner(batch)

        # Project the encoder outputs to avoid
        # unnecessary matmuls in the decoder loop
        encoder_seq = self.encoder(x)
        if self.speaker_emb_dim > 0:
            speaker_emb = speaker_emb[:, None, :]
            speaker_emb = speaker_emb.repeat(1, encoder_seq.shape[1], 1)
            encoder_seq = torch.cat([encoder_seq, speaker_emb], dim=2)

        encoder_seq = torch.cat([encoder_seq, aligner_out['x'].detach()], dim=2)

        encoder_seq_proj_query = self.encoder_proj_query(encoder_seq)
        encoder_seq_proj = self.encoder_proj(encoder_seq)

        mel_outputs, attn_scores = [], []

        mel_in = torch.cat([mel, aligner_out['mel'].transpose(1, 2).detach()], dim=1)

        # Run the decoder loop
        for t in range(0, steps, self.r):
            prenet_in = mel_in[:, :, t - 1] if t > 0 else go_frame
            mel_frames, scores, hidden_states, cell_states, context_vec = \
                self.decoder(encoder_seq_proj_query, encoder_seq_proj, prenet_in,
                             hidden_states, cell_states, context_vec, t)
            mel_outputs.append(mel_frames)
            attn_scores.append(scores)

        # Concat the mel outputs into sequence
        mel_outputs = torch.cat(mel_outputs, dim=2)

        # Post-Process for Linear Spectrograms
        postnet_out = self.postnet(mel_outputs)
        linear = self.post_proj(postnet_out)
        linear = linear.transpose(1, 2)

        # For easy visualisation
        attn_scores = torch.cat(attn_scores, 1)
        # attn_scores = attn_scores.cpu().data.numpy()

        return {'mel': mel_outputs, 'mel_post': linear, 'att': attn_scores, 'att_aligner': aligner_out['att']}


    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        # assignment to parameters or buffers is overloaded, updates internal dict entry
        self.step = self.step.data.new_tensor(1)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Tacotron':
        model_config = config['tacotron']['model']
        model_config['num_chars'] = len(phonemes)
        model_config['n_mels'] = config['dsp']['num_mels']
        return Tacotron(**model_config)

    @classmethod
    def from_checkpoint(cls, path: Union[Path, str]) -> 'Tacotron':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = Tacotron.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        return model