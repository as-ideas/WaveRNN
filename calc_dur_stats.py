import pandas as pd
from pathlib import Path
import numpy as np

def running_median(arr):
    result = np.zeros_like(arr, dtype=float)

    for i in range(0, len(arr)):
        window_slice = arr[i: i + 7]
        result[i] = np.median(window_slice)

    return result


if __name__ == '__main__':

    dur_files = list(Path('/Users/cschaefe/datasets/multispeaker_welt_bild/alg').glob('**/*.npy'))
    welt_durs = []
    bild_durs = []

    for dur_file in dur_files[:1000]:
        id = ''

        dur = np.load(str(dur_file))

        med = running_median(dur)
        print(dur)
        print(med)

        if 'welt_' in dur_file.stem:
            welt_durs.append(dur)
        else:
            bild_durs.append(dur)

    exit()
    welt_durs_cat = np.concatenate(welt_durs)
    bild_durs_cat = np.concatenate(bild_durs)

    welt_mean, welt_std = np.mean(welt_durs_cat), np.std(welt_durs_cat)

    for dur in welt_durs:
        dur_norm = (dur - welt_mean) / welt_std
        print(dur)
        print(dur_norm)
        print(np.log(dur_norm))