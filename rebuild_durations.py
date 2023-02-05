import random
import shutil
from collections import Counter
from itertools import groupby
from pathlib import Path
import librosa
import torch
import tqdm
from dp.utils.io import pickle_binary
from numpy.polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch import optim

from models.tacotron import Tacotron
from trainer.common import np_now
from utils.checkpoints import restore_checkpoint
from utils.display import plot_pitch, plot_mel, plot_attention
from utils.dsp import DSP
from utils.duration_extractor import DurationExtractor
from utils.files import unpickle_binary, read_config
from utils.metrics import attention_score
from utils.paths import Paths
from utils.text.tokenizer import Tokenizer

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


if __name__ == '__main__':
    config = read_config('config.yaml')
    dsp = DSP.from_config(config)
    paths = Paths(config['data_path'], config['tts_model_id'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # Instantiate Tacotron Model
    print('\nInitialising Tacotron Model...\n')
    model = Tacotron.from_config(config).to(device)
    model.eval()
    model.decoder.prenet.train()

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint(model=model, optim=optimizer,
                       path=paths.taco_checkpoints / 'latest_model.pt',
                       device=device)

    dsp = DSP.from_config(config)
    stds = []
    text_dict = unpickle_binary('data_bild/text_dict.pkl')
    id = 'bild_r_0036_021'
    tokenizer = Tokenizer()
    duration_extractor = DurationExtractor(silence_threshold=-11, silence_prob_shift=0.25)
    att_score_dict = {}
    sum_att_score = 0
    att_score_dict_orig = unpickle_binary(paths.data / 'att_score_dict.pkl')
    num_failed = 0
    for i, id in enumerate(tqdm.tqdm(text_dict.keys(), total=len(text_dict), smoothing=0.1), 1):
        try:
            wav = dsp.load_wav(f'/Users/cschaefe/datasets/tts-synth-data/bild/bild_snippets/{id}.wav')

            wav = dsp.trim_long_silences(wav)

            mel = torch.from_numpy(dsp.wav_to_mel(wav)).unsqueeze(0)

            mel_mask = dsp.get_mel_mask(wav, mel, max_sil_len=1)

            #print(id, mel_mask.shape, mel.shape)
            mel_trimmed = mel[:, :, mel_mask]

            split_points = zero_runs(mel_mask)

            #print(split_points)
            x = text_dict[id]
            x = torch.tensor(tokenizer(x)).unsqueeze(0).long()
            #_, _, att_orig = model(x, mel_orig)
            with torch.no_grad():
                _, _, att = model(x, mel_trimmed)
            #plt.clf()
            #plot_attention(att.squeeze().detach().numpy())
            #plt.savefig(f'/tmp/att/{id}_att.png')
            #plt.close()

            parts = []
            last_b = 0
            piece = 0
            last_piece = 0
            for a, b in split_points:
                piece += a - last_b
                last_b = b
                zeros = torch.zeros((1, b-a, att.size(-1)))
                parts.append(att[:, last_piece:piece:, :])
                parts.append(zeros)
                diff = b-a
                last_a = a - diff
                last_piece = piece


            parts.append(att[:, last_piece:, :])

            att_new = torch.cat(parts, dim=1)

            mel[:, :, ~mel_mask] = -12
            align_score, _ = attention_score(att, torch.ones(1)*(mel_trimmed.shape[-1]), r=1)
            align_score = float(align_score[0])
            durs, att_score = duration_extractor(x=x.squeeze(), mel=mel.squeeze(), att=att_new.squeeze())
            durs = np_now(durs).astype(np.int)
            att_score_dict[id] = (align_score, att_score)
            sum_att_score += att_score

            assert sum(durs) == mel.shape[-1]
            mean_att_score = sum_att_score / i
            np.save(paths.alg / f'{id}.npy', np.array(durs).astype(int))
            np.save(paths.mel / f'{id}.npy', mel)
            att_score_orig = att_score_dict_orig[id][1]
            print('mean att_score: ', mean_att_score, ' att score: ', att_score_orig, att_score, ' successful: ', i-num_failed, ' failed: ', num_failed)
            pickle_binary(att_score_dict, paths.data / 'att_score_dict_reextracted.pkl')
            durs_orig = np.load(f'data_bild/alg/{id}.npy')
            print(durs_orig.tolist())
            print(durs.tolist())

            for t, d1, d2 in zip(text_dict[id], durs_orig, durs):
                print(t, d1, d2)
        except Exception as e:
            num_failed += 1
            print(e)

    pickle_binary(att_score_dict, paths.data / 'att_score_dict_reextracted.pkl')


    """
    x = text_dict[id]
    x = torch.tensor(tokenizer(x)).unsqueeze(0).long()



    plot_attention(att.squeeze().detach().numpy())

    #plot_mel(mel.squeeze().numpy())
    #plt.savefig(f'/tmp/att/{id}_mel_trim.png')
    plt.savefig(f'/tmp/att/{id}_trim.png')
    """