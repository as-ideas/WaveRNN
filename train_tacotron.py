import argparse
import itertools
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader

from duration_extraction.duration_extraction_pipe import DurationExtractionPipeline
from duration_extraction.duration_extractor import DurationExtractor
from models.tacotron import Tacotron
from trainer.common import to_device
from trainer.taco_trainer import TacoTrainer
from utils.checkpoints import restore_checkpoint
from utils.dataset import get_tts_datasets
from utils.display import *
from utils.dsp import DSP
from utils.files import pickle_binary, unpickle_binary, read_config
from utils.paths import Paths


def normalize_values(phoneme_val) -> Tuple[float, float]:
    """
    Normalizes non-zero phoneme values to zero mean and unitary
    variance and returns mean and std of the original distribution.
    """
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]]
                               for item_id, v in phoneme_val])
    mean, std = np.mean(nonzeros), np.std(nonzeros)
    for item_id, v in phoneme_val:
        zero_idxs = np.where(v == 0.0)[0]
        v -= mean
        v /= std
        v[zero_idxs] = 0.0
    return float(mean), float(std)


# adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/
# 0b27e359a5869cd23294c1707c92f989c0bf201e/PyTorch/SpeechSynthesis/FastPitch/extract_mels.py
def extract_pitch_energy(save_path_pitch: Path,
                         save_path_energy: Path,
                         pitch_max_freq: float) -> Tuple[float, float]:
    train_data = unpickle_binary(paths.data / 'train_dataset.pkl')
    val_data = unpickle_binary(paths.data / 'val_dataset.pkl')
    all_data = train_data + val_data
    phoneme_pitches = []
    phoneme_energies = []
    for prog_idx, (item_id, mel_len) in enumerate(all_data, 1):
        dur = np.load(paths.alg / f'{item_id}.npy')
        mel = np.load(paths.mel / f'{item_id}.npy')
        energy = np.linalg.norm(np.exp(mel), axis=0, ord=2)
        assert np.sum(dur) == mel_len
        pitch = np.load(paths.raw_pitch / f'{item_id}.npy')
        durs_cum = np.cumsum(np.pad(dur, (1, 0)))
        pitch_char = np.zeros((dur.shape[0],), dtype=np.float32)
        energy_char = np.zeros((dur.shape[0],), dtype=np.float32)
        for idx, a, b in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
            values = pitch[a:b][np.where(pitch[a:b] != 0.0)[0]]
            values = values[np.where(values < pitch_max_freq)[0]]
            pitch_char[idx] = np.mean(values) if len(values) > 0 else 0.0
            energy_values = energy[a:b]
            energy_char[idx] = np.mean(energy_values)if len(energy_values) > 0 else 0.0
        phoneme_pitches.append((item_id, pitch_char))
        phoneme_energies.append((item_id, energy_char))
        bar = progbar(prog_idx, len(all_data))
        msg = f'{bar} {prog_idx}/{len(all_data)} Files '
        stream(msg)

    for item_id, phoneme_energy in phoneme_energies:
        np.save(str(save_path_energy / f'{item_id}.npy'), phoneme_energy, allow_pickle=False)

    mean, var = normalize_values(phoneme_pitches)
    for item_id, phoneme_pitch in phoneme_pitches:
        np.save(str(save_path_pitch / f'{item_id}.npy'), phoneme_pitch, allow_pickle=False)

    print(f'\nPitch mean: {mean} var: {var}')

    return mean, var


def create_gta_features(model: Tacotron,
                        train_set: DataLoader,
                        val_set: DataLoader,
                        save_path: Path):
    model.eval()
    device = next(model.parameters()).device  # use same device as model parameters
    iters = len(train_set) + len(val_set)
    dataset = itertools.chain(train_set, val_set)
    for i, batch in enumerate(dataset, 1):
        batch = to_device(batch, device=device)
        with torch.no_grad():
            _, gta, _ = model(batch['x'], batch['mel'])
        gta = gta.cpu().numpy()
        for j, item_id in enumerate(batch['item_id']):
            mel = gta[j][:, :batch['mel_len'][j]]
            np.save(str(save_path/f'{item_id}.npy'), mel, allow_pickle=False)
        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


def create_align_features(model: Tacotron,
                          paths: Paths,
                          config: Dict[str, Any]) -> None:

    assert model.r == 1, f'Reduction factor of tacotron must be 1 for creating alignment features! ' \
                         f'Reduction factor was: {model.r}'
    model.eval()
    model.decoder.prenet.train()

    dur_extr_conf = config['duration_extraction']

    duration_extractor = DurationExtractor(silence_threshold=dur_extr_conf['silence_threshold'],
                                           silence_prob_shift=dur_extr_conf['silence_prob_shift'])

    duration_extraction_pipe = DurationExtractionPipeline(paths=paths, config=config,
                                                          duration_extractor=duration_extractor)

    print('Extracting attention matrices from tacotron...')
    duration_extraction_pipe.extract_attentions(model, max_batch_size=dur_extr_conf['max_batch_size'])

    num_workers = dur_extr_conf['num_workers']
    print(f'Extracting durations from attention matrices (num workers={num_workers})...')
    att_score_dict = duration_extraction_pipe.extract_durations(num_workers=num_workers,
                                                                sampler_bin_size=num_workers*4)
    pickle_binary(att_score_dict, paths.data / 'att_score_dict.pkl')

    print('Extracting Pitch Values...')
    extract_pitch_energy(save_path_pitch=paths.phon_pitch,
                         save_path_energy=paths.phon_energy,
                         pitch_max_freq=config['dsp']['pitch_max_freq'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_align', '-a', action='store_true', help='Force the model to create attention alignment features')
    parser.add_argument('--extract_pitch', '-p', action='store_true', help='Extracts phoneme-pitch values only')
    parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams.')

    args = parser.parse_args()
    config = read_config(args.config)
    dsp = DSP.from_config(config)
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])

    if args.extract_pitch:
        print('Extracting Pitch and Energy Values...')
        mean, var = extract_pitch_energy(save_path_pitch=paths.phon_pitch,
                                         save_path_energy=paths.phon_energy,
                                         pitch_max_freq=dsp.pitch_max_freq)
        print('\n\nYou can now train ForwardTacotron - use python train_forward.py\n')
        exit()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # Instantiate Tacotron Model
    print('\nInitialising Tacotron Model...\n')
    model = Tacotron.from_config(config).to(device)

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint(model=model, optim=optimizer,
                       path=paths.taco_checkpoints / 'latest_model.pt',
                       device=device)

    train_cfg = config['tacotron']['training']
    if args.force_gta:
        print('Creating Ground Truth Aligned Dataset...\n')
        train_set, val_set = get_tts_datasets(paths.data, 1, model.r,
                                              max_mel_len=train_cfg['max_mel_len'],
                                              filter_attention=False)
        create_gta_features(model, train_set, val_set, paths.gta)
        print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')
    elif args.force_align:
        print('Creating Attention Alignments and Pitch Values...')
        train_set, val_set = get_tts_datasets(paths.data, 1, model.r,
                                              max_mel_len=None,
                                              filter_attention=False)
        create_align_features(model=model, config=config,  paths=paths)
        print('\n\nYou can now train ForwardTacotron - use python train_forward.py\n')
    else:
        trainer = TacoTrainer(paths, config=config, dsp=dsp)
        trainer.train(model, optimizer)
        print('Training finished, now creating Attention Alignments and Pitch Values...')
        train_set, val_set = get_tts_datasets(paths.data, 1, model.r,
                                              max_mel_len=None,
                                              filter_attention=False)
        create_align_features(model=model,  config=config, paths=paths)
        print('\n\nYou can now train ForwardTacotron - use python train_forward.py\n')









