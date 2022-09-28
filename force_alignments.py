

import itertools

import torch
from torch import optim
from tqdm import tqdm

from models.tacotron import Tacotron
from trainer.common import to_device
from utils.checkpoints import restore_checkpoint
from utils.dataset import get_tts_datasets
from utils.display import *
from utils.dsp import DSP
from utils.duration_extractor import DurationExtractor
from utils.files import read_config
from utils.metrics import attention_score
from utils.paths import Paths

if __name__ == '__main__':
    config = read_config('config.yaml')
    dsp = DSP.from_config(config)
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])

    print('\nInitialising Tacotron Model...\n')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Tacotron.from_config(config).to(device)

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint(model=model, optim=optimizer,
                       path=paths.taco_checkpoints / 'latest_model.pt',
                       device=device)

    model.eval()
    model.decoder.prenet.train()
    duration_extractor = DurationExtractor(
        silence_threshold=config['preprocessing']['silence_threshold'],
        silence_prob_shift=config['preprocessing']['silence_prob_shift'])

    BATCH_SIZE = 1

    train_set, val_set = get_tts_datasets(paths.data, BATCH_SIZE, model.r,
                                          max_mel_len=None,
                                          filter_attention=False)
    dataset = itertools.chain(train_set, val_set)
    sum_att_score = 0

    print('Performing model inference...')
    pbar = tqdm(dataset, total=len(val_set)+len(train_set))
    for i, batch in enumerate(pbar, 1):
        pbar.set_description(f'Avg attention score: {sum_att_score / (i * BATCH_SIZE)}', refresh=True)
        batch = to_device(batch, device=device)
        with torch.no_grad():
            _, _, att_batch = model(batch['x'], batch['mel'], batch['speaker_emb'])
        for b in range(BATCH_SIZE):
            x = batch['x'][b].cpu()
            x_len = batch['x_len'][b].cpu()
            mel_len = batch['mel_len'][b].cpu()
            item_id = batch['item_id'][b]
            att = att_batch[b, :mel_len, :x_len].cpu()
            _, att_score = attention_score(att_batch, batch['mel_len'], r=1)
            sum_att_score += att_score.sum()
            np.save(paths.att_pred / f'{item_id}.npy', att.numpy(), allow_pickle=False)


