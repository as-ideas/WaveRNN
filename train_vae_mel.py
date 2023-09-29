from tensorboardX import SummaryWriter

from trainer.common import MaskedL1, to_device
from utils.dataset import get_forward_dataloaders
from utils.files import read_config
from utils.paths import Paths
from torch import nn
import torch.nn.functional as F
import torch

from utils.text.symbols import phonemes

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class VAEPredictor(nn.Module):

    def __init__(self):
        super(VAEPredictor, self).__init__()
        self.rnn_m = nn.GRU(80, 64, bidirectional=True, batch_first=True)
        self.rnn_v = nn.GRU(80, 64, bidirectional=True, batch_first=True)
        self.out_conv = nn.Conv1d(128, 256, 1, padding=0)
        self.out_conv2 = nn.Conv1d(256, 80, 1, padding=0)

    def forward(self, x):
        z_mean = F.leaky_relu(self.rnn_m(x.transpose(1, 2))[0], negative_slope=0.2).transpose(1, 2)
        z_log_var = F.leaky_relu(self.rnn_v(x.transpose(1, 2))[0], negative_slope=0.2).transpose(1, 2)
        noise = torch.rand_like(z_log_var).to(x.device)
        z = z_mean + torch.exp(0.5 * z_log_var) * noise
        out = F.leaky_relu(self.out_conv(z), negative_slope=0.2)
        out = self.out_conv2(out)
        return out, z, z_mean, z_log_var


class PredModel(nn.Module):

    def __init__(self):
        super(PredModel, self).__init__()
        self.emb = nn.Embedding(num_embeddings=len(phonemes), embedding_dim=64)
        self.rnn_m = nn.GRU(80, 64, bidirectional=True, batch_first=True)
        self.rnn_v = nn.GRU(80, 64, bidirectional=True, batch_first=True)
        self.out_conv = nn.Conv1d(128, 256, 1, padding=0)
        self.out_conv2 = nn.Conv1d(256, 80, 1, padding=0)

    def forward(self, x):
        z_mean = F.leaky_relu(self.rnn_m(x)[0], negative_slope=0.2).transpose(1, 2)
        z_log_var = F.leaky_relu(self.rnn_v(x)[0], negative_slope=0.2).transpose(1, 2)
        noise = torch.rand_like(z_log_var).to(x.device)
        z = z_mean + torch.exp(0.5 * z_log_var) * noise
        out = F.leaky_relu(self.out_conv(z), negative_slope=0.2)
        out = self.out_conv2(out)

        return out, z, z_mean, z_log_var



if __name__ == '__main__':

    config = read_config('configs/multispeaker.yaml')
    filter_params = config['multi_forward_tacotron']['training']['filter']

    paths = Paths(config['data_path'], config['tts_model_id'])

    train_set, val_set = get_forward_dataloaders(paths=paths, batch_size=32, **filter_params)

    model = VAEPredictor().to(DEVICE)
    model2 = VAEPredictor().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    optim2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    masked_l1_loss = MaskedL1()

    sw = SummaryWriter('logs_vae')

    step = 0

    for epoch in range(10):
        for i, batch in enumerate(train_set):
            batch = to_device(batch, DEVICE)
            out_q, z_q, m_q, logs_q = model(batch['mel'])
            out_p, z_p, m_p, logs_p = model2(batch['mel'])

            with torch.no_grad():
                out2 = F.leaky_relu(model.out_conv(z_p), negative_slope=0.2)
                out2 = model.out_conv2(out2)

            l1_loss = masked_l1_loss(out_q, batch['mel'], batch['mel_len'])
            l1_loss_2 = masked_l1_loss(out2, batch['mel'], batch['mel_len']).detach()

            #kl_loss_2 = - 0.5 * torch.mean(1 + logs_p - m_p.pow(2) - logs_p.exp())

            #kl_loss = - 0.5 * torch.mean(1 + logs_q - m_q.pow(2) - logs_q.exp())
            #kl_diff_loss = logs_p - logs_q - 0.5 + (logs_q.exp().pow(2) + (m_p - m_q).pow(2)) / (2 * logs_p.exp().pow(2))
            # kl_loss = - 0.5 * torch.sum(1+ logs_q - m_q.pow(2) - logs_q.exp())
            kl_loss = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
            kl_loss = kl_loss.mean()
            #kl_diff_loss = kl_diff_loss.mean()

            loss = kl_loss +  l1_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            step += 1

            sw.add_scalar('kl_loss', kl_loss, step)
            sw.add_scalar('l1_loss', l1_loss, step)
            sw.add_scalar('l1_loss_2', l1_loss_2, step)
            sw.add_scalar('q_mean', torch.mean(m_q), step)
            sw.add_scalar('q_var', torch.mean(logs_q.exp()), step)

            print(i, float(l1_loss), float(l1_loss_2), 'kl_loss', float(kl_loss), 'mean', float(torch.mean(m_q)), 'std', float(torch.mean(torch.exp(logs_q))))
