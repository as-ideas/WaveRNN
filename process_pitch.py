import numpy as np
from pycwt import wavelet
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



def get_lf0_cwt(lf0):
    """
    input:
        signal of shape (N)
    output:
        Wavelet_lf0 of shape(10, N), scales of shape(10)
    """
    mother = wavelet.MexicanHat()
    dt = 0.005
    dj = 1
    s0 = dt * 2
    J = 9

    Wavelet_lf0, scales, _, _, _, _ = wavelet.cwt(np.squeeze(lf0), dt, dj, s0, J, mother)
    # Wavelet.shape => (J + 1, len(lf0))
    Wavelet_lf0 = np.real(Wavelet_lf0).T
    return Wavelet_lf0, scales


if __name__ == '__main__':

    config = read_config('configs/multispeaker.yaml')
    paths = Paths(config['data_path'], config['tts_model'])

    text_dict = unpickle_binary(paths.text_dict)

    for id in tqdm(text_dict.keys(), total=len(text_dict)):
        pitch = np.load(paths.phon_pitch / f'{id}.npy')
#        pitch = convert_continuos_f0(pitch)
#        pitch = (pitch - np.mean(pitch) / np.std(pitch))
        w, s = get_lf0_cwt(pitch)
        w = np.transpose(w, (1, 0))

        #if len(pitch) >= 300:
        #    print('skipped', id, len(pitch))
        #    continue
        #pitch_padded = np.pad(pitch, (0, 300 - len(pitch)), mode='constant')
        #pitch_wave = cwt(pitch)
        #pitch_wave = np.abs(pitch_wave[0])[:, :len(pitch)]

        #plt.clf()
        #plot_mel(w)
        #plt.savefig(f'/tmp/pitch/{id}.png')

        #plt.clf()
        #plot_pitch(pitch)
        #plt.savefig(f'/tmp/pitch/{id}_pitch.png')

        #dur = np.load(paths.alg / f'{id}.npy')
        #dur_padded = np.pad(dur, (0, 300 - len(dur)), mode='constant')
        #dur_wave = cwt(dur_padded)
        #dur_wave = np.abs(dur_wave[0])[:, :len(dur)]

        np.save(paths.pitch_cwt / f'{id}.npy', w)
        #np.save(paths.alg_cwt / f'{id}.npy', dur_wave)
