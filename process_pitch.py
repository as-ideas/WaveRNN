import numpy as np
from scipy.interpolate import interp1d
from ssqueezepy import cwt, icwt
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.display import plot_mel, plot_pitch
from utils.files import read_config, unpickle_binary
from utils.paths import Paths


def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0
    Args:
        f0 (ndarray): original f0 sequence with the shape (T)
    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    f0 = np.copy(f0)
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        print("| all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return cont_f0

if __name__ == '__main__':

    config = read_config('configs/multispeaker.yaml')
    paths = Paths(config['data_path'], config['tts_model'])

    text_dict = unpickle_binary(paths.text_dict)

    for id in tqdm(text_dict.keys(), total=len(text_dict)):
        pitch = np.load(paths.alg / f'{id}.npy')
        if len(pitch) >= 300:
            print('skipped', id, len(pitch))
            continue
        pitch_padded = np.pad(pitch, (0, 300 - len(pitch)), mode='constant')
        pitch_wave = cwt(pitch_padded)
        pitch_wave = np.abs(pitch_wave[0])[:, :len(pitch)]

        dur = np.load(paths.alg / f'{id}.npy')
        dur_padded = np.pad(dur, (0, 300 - len(dur)), mode='constant')
        dur_wave = cwt(dur_padded)
        dur_wave = np.abs(dur_wave[0])[:, :len(dur)]

        np.save(paths.pitch_cwt / f'{id}.npy', pitch_wave)
        np.save(paths.alg_cwt / f'{id}.npy', dur_wave)
