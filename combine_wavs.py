from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.display import plot_mel

if __name__ == '__main__':


    mel_paths = Path('/tmp/mel/').glob('*npy')

    for mel_path in mel_paths:

        mel = np.load(str(mel_path))

        fig = plot_mel(mel)
        plt.savefig(f'/tmp/audio/{mel_path.stem}.png')
        plt.close()