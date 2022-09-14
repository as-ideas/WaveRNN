import argparse
import itertools
import os
import subprocess
from pathlib import Path
from typing import Union

import torch
import tqdm
from torch import optim
from torch.nn import init
from torch.utils.data.dataloader import DataLoader

from models.forward_tacotron import ForwardTacotron
from models.tacotron import Tacotron
from trainer.common import to_device
from trainer.forward_trainer import ForwardTrainer
from utils.checkpoints import restore_checkpoint, init_tts_model
from utils.dataset import get_tts_datasets
from utils.display import *
from utils.dsp import DSP
from utils.files import read_config, unpickle_binary
from utils.paths import Paths


def try_get_git_hash() -> Union[str, None]:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        print(f'Could not retrieve git hash! {e}')
        return None


def create_gta_features(model: Tacotron,
                        train_set: DataLoader,
                        val_set: DataLoader,
                        save_path: Path) -> None:
    model.eval()
    device = next(model.parameters()).device  # use same device as model parameters
    iters = len(train_set) + len(val_set)
    dataset = itertools.chain(train_set, val_set)
    for i, batch in enumerate(dataset, 1):
        batch = to_device(batch, device=device)

        with torch.no_grad():
            pred = model(batch)
        gta = pred['mel_post'].cpu().numpy()
        for j, item_id in enumerate(batch['item_id']):
            mel = gta[j][:, :batch['mel_len'][j]]
            np.save(str(save_path/f'{item_id}.npy'), mel, allow_pickle=False)
        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ForwardTacotron TTS')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams.')
    args = parser.parse_args()

    config = read_config(args.config)
    if 'git_hash' not in config or config['git_hash'] is None:
        config['git_hash'] = try_get_git_hash()
    dsp = DSP.from_config(config)
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])

    assert len(os.listdir(paths.alg)) > 0, f'Could not find alignment files in {paths.alg}, please predict ' \
                                           f'alignments first with python train_tacotron.py --force_align!'

    force_gta = args.force_gta
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # Instantiate Forward TTS Model
    speaker_dict = unpickle_binary(paths.data / 'speaker_dict.pkl')
    speaker_names = {s for s in speaker_dict.values() if len(s) > 1}
    config['speaker_names'] = speaker_names
    model = init_tts_model(config).to(device)
    print(f'\nInitialized tts model: {model}\n')

    print('Loading semb')
    sembs = list(paths.speaker_emb.glob('*.npy'))

    print(f'Speaker names: {speaker_names}')
    speaker_emb = {name: np.zeros(256) for name in speaker_names}
    speaker_norm = {name: 0. for name in speaker_names}

    for f in tqdm.tqdm(sembs, total=len(sembs)):
        try:
            item_id = f.stem
            speaker_name = speaker_dict[item_id]
            emb = np.load(paths.speaker_emb / f'{item_id}.npy')
            speaker_emb[speaker_name] += emb
            speaker_norm[speaker_name] += 1
        except Exception as e:
            print(e)


    optimizer = optim.Adam(model.parameters())
    restore_checkpoint(model=model, optim=optimizer,
                       path=paths.forward_checkpoints / 'latest_model.pt',
                       device=device)

    for speaker_name in tqdm.tqdm(speaker_names, total=len(speaker_names)):
        print(speaker_name)
        print(speaker_emb[speaker_name])
        print(speaker_norm[speaker_name])
        emb = speaker_emb[speaker_name] / speaker_norm[speaker_name]
        emb = torch.tensor(emb).float().to(device)
        print(emb)

        setattr(model, speaker_name, emb)

    if force_gta:
        print('Creating Ground Truth Aligned Dataset...\n')
        train_set, val_set = get_tts_datasets(
            paths.data, 8, r=1, model_type='forward',
            filter_attention=False, max_mel_len=None)
        create_gta_features(model, train_set, val_set, paths.gta)
        print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')
    else:
        trainer = ForwardTrainer(paths=paths, dsp=dsp, config=config)
        trainer.train(model, optimizer)

