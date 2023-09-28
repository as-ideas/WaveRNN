from trainer.common import MaskedL1
from utils.dataset import get_forward_dataloaders
from utils.files import read_config
from utils.paths import Paths
from torch import nn
import torch.nn.functional as F
import torch


class VAEPredictor(nn.Module):

    def __init__(self):
        super(VAEPredictor, self).__init__()
        self.conv_m = nn.GRU(80, 64, bidirectional=True, batch_first=True)
        self.conv_v = nn.GRU(80, 64, bidirectional=True, batch_first=True)
        self.out_conv = nn.Conv1d(128, 256, 1, padding=0)
        self.out_conv2 = nn.Conv1d(256, 80, 1, padding=0)

    def forward(self, x):
        z_mean = F.leaky_relu(self.conv_m(x.transpose(1, 2))[0], negative_slope=0.2).transpose(1, 2)
        z_log_var = F.leaky_relu(self.conv_v(x.transpose(1, 2))[0], negative_slope=0.2).transpose(1, 2)
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

    model = VAEPredictor()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    masked_l1_loss = MaskedL1()

    for i, batch in enumerate(train_set):

        out, z, z_mean, z_log = model(batch['mel'])

        l1_loss = masked_l1_loss(out, batch['mel'], batch['mel_len'])

        kl_diff_loss = 0
        B = out.size(0)

        kl_loss = - 0.5 * torch.mean(1 + z_log - z_mean.pow(2) - z_log.exp())

        loss = kl_loss + l1_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        print(i, float(l1_loss), float(kl_loss), float(torch.mean(z_mean)), float(torch.mean(torch.exp(z_log))))