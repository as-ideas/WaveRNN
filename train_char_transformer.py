import math
import random

import pandas as pd
# Define the dataset
import torch
import tqdm
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.text.symbols import phonemes
from utils.text.tokenizer import Tokenizer


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


def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, None)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


if __name__ == '__main__':

    df = pd.read_csv('/Users/cschaefe/datasets/nlp/welt_articles_phonemes.tsv', sep='\t', encoding='utf-8')
    df.dropna(inplace=True)
    strings = df['phonemes']
    strings = [s for s in strings if len(s) > 10]
    print(strings[:10])

    dataset = StringDataset(strings)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    device = torch.device('mps')#torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ntokens = len(phonemes) + 1  # size of vocabulary
    mask_index = len(phonemes)
    emsize = 512  # embedding dimension
    d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    step = 0

    eval_sent = 'diː t͡seː-deː-ʔuː-t͡sɛntʁaːlə lɛst iːɐ ʃpɪt͡sn̩pɛʁzonaːl dʊʁçt͡ʃɛkn̩: ɪm jʏŋstn̩ mɪtɡliːdɐbʁiːf fɔn bʊndəsɡəʃɛft͡sfyːʁɐ ʃtɛfan hɛnəvɪç (axtʔʊntfɪʁt͡sɪç) vɪʁt diː t͡seː-deː-ʔuː-baːzɪs aʊfɡəfɔʁdɐt, an aɪnɐ bəfʁaːɡʊŋ dɛs tʁiːʁɐ paʁtaɪən-fɔʁʃɐs uːvə jan (nɔɪnʔʊntfʏnft͡sɪç) taɪlt͡suneːmən'
    eval_input = Tokenizer()(eval_sent)
    eval_tens = torch.tensor(eval_input).unsqueeze(0).to(device)
    eval_mask = (torch.rand(eval_tens.size()) < 0.3).to(device)
    eval_tens[eval_mask] = mask_index

    sw = SummaryWriter(log_dir='checkpoints/lm_summary')

    for epoch in range(100):
        for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
            batch = batch.to(device)
            batch_target = torch.clone(batch)
            rand_mask = (torch.rand(batch.size()) < 0.3).to(device)

            batch[rand_mask] = mask_index

            output = model(batch)

            loss = criterion(output.transpose(1, 2), batch_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            step += 1
            sw.add_scalar('masked_loss', loss, global_step=step)
            if step % 100 == 0:
                out = model(eval_tens)
                out = torch.argmax(out[0], dim=-1)
                out_text = Tokenizer().decode(out.tolist())
                sw.add_text('eval/target', '"""' + eval_sent + '"""', global_step=step)
                sw.add_text('eval/pred', '"""' + out_text + '"""', global_step=step)
                eval_score = 0
                for a, b in zip(out_text, eval_sent):
                    if a == b:
                        eval_score += 1
                sw.add_scalar('eval/score', eval_score / len(eval_sent), global_step=step)
                print(step, loss, eval_score / len(eval_sent))
                print(eval_sent)
                print(out_text)

                torch.save(model.state_dict(), 'checkpoints/latest_language_model.pt')
