import argparse
import itertools
from pathlib import Path

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from models.tacotron import Tacotron
from trainer.taco_trainer import TacoTrainer
from utils import hparams as hp
from utils.checkpoints import restore_checkpoint
from utils.dataset import get_tts_datasets
from utils.display import *
from utils.dsp import np_now
from utils.paths import Paths
from utils.text import phonemes


def create_gta_features(model: Tacotron,
                        train_set: DataLoader,
                        val_set: DataLoader,
                        save_path: Path):
    model.eval()
    device = next(model.parameters()).device  # use same device as model parameters
    iters = len(train_set) + len(val_set)
    dataset = itertools.chain(train_set, val_set)
    for i, (x, mels, ids, mel_lens) in enumerate(dataset, 1):
        x, mels = x.to(device), mels.to(device)
        with torch.no_grad():
            _, gta, _ = model(x, mels)
        gta = gta.cpu().numpy()
        for j, item_id in enumerate(ids):
            mel = gta[j][:, :mel_lens[j]]
            np.save(str(save_path/f'{item_id}.npy'), mel, allow_pickle=False)
        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


def create_align_features(model: Tacotron,
                          train_set: DataLoader,
                          val_set: DataLoader,
                          save_path: Path):
    assert model.r == 1, f'Reduction factor of tacotron must be 1 for creating alignment features! ' \
                         f'Reduction factor was: {model.r}'
    model.eval()
    device = next(model.parameters()).device  # use same device as model parameters
    iters = len(val_set) + len(train_set)
    dataset = itertools.chain(train_set, val_set)
    for i, (x, mels, ids, mel_lens) in enumerate(dataset, 1):
        x, mels = x.to(device), mels.to(device)
        with torch.no_grad():
            _, _, attn = model(x, mels)
        attn = np_now(attn)
        bs, chars = attn.shape[0], attn.shape[2]
        argmax = np.argmax(attn[:, :, :], axis=2)
        mel_counts = np.zeros(shape=(bs, chars), dtype=np.int32)
        for b in range(attn.shape[0]):
            # fix random jumps in attention 
            for j in range(1, argmax.shape[1]):
                if abs(argmax[b, j] - argmax[b, j-1]) > 10:
                    argmax[b, j] = argmax[b, j-1]
            count = np.bincount(argmax[b, :mel_lens[b]])
            mel_counts[b, :len(count)] = count[:len(count)]

        for j, item_id in enumerate(ids):
            np.save(str(save_path / f'{item_id}.npy'), mel_counts[j, :], allow_pickle=False)
        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_align', '-a', action='store_true', help='Force the model to create attention alignment features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    force_train = args.force_train
    force_gta = args.force_gta
    force_align = args.force_align

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Tacotron Model
    print('\nInitialising Tacotron Model...\n')
    model = Tacotron(embed_dims=hp.tts_embed_dims,
                     num_chars=len(phonemes),
                     encoder_dims=hp.tts_encoder_dims,
                     decoder_dims=hp.tts_decoder_dims,
                     n_mels=hp.num_mels,
                     fft_bins=hp.num_mels,
                     postnet_dims=hp.tts_postnet_dims,
                     encoder_K=hp.tts_encoder_K,
                     lstm_dims=hp.tts_lstm_dims,
                     postnet_K=hp.tts_postnet_K,
                     num_highways=hp.tts_num_highways,
                     dropout=hp.tts_dropout,
                     stop_threshold=hp.tts_stop_threshold).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Num Params: {params}')

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('tts', paths, model, optimizer, create_if_missing=True, device=device)

    if force_gta:
        print('Creating Ground Truth Aligned Dataset...\n')
        train_set, val_set = get_tts_datasets(paths.data, 8, model.r)
        create_gta_features(model, train_set, val_set, paths.gta)
        print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')
    elif force_align:
        print('Creating Attention Alignments...\n')
        train_set, val_set = get_tts_datasets(paths.data, 8, model.r)
        create_align_features(model, train_set, val_set, paths.alg)
        print('\n\nYou can now train ForwardTacotron - use python train_forward.py\n')
    else:
        trainer = TacoTrainer(paths)
        trainer.train(model, optimizer)









