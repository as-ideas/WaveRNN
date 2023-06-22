from pathlib import Path
import torch
import numpy as np
from utils.files import unpickle_binary

device = torch.device('cpu')


hubert_files = Path('/Users/cschaefe/datasets/multispeaker_welt_bild').glob('**/*.pt')


text_dict = unpickle_binary('/Users/cschaefe/datasets/multispeaker_welt_bild/text_dict.pkl')

for hub in hubert_files:
    hubert = torch.load(hub)
    id = hub.stem
    durs = np.load(f'/Users/cschaefe/datasets/multispeaker_welt_bild/alg/{id}.npy')
    hub_fact = hubert.size(1) / durs.sum()
    dur_hub = torch.zeros((int(durs.shape[0]), 1024))
    dur_ind = 0
    for dur_ind, dur in enumerate(durs):
        for i in range(dur):
            hub_ind = min(int(hub_fact * dur_ind), hubert.size(1)-1)
            dur_hub[dur_ind] += hubert[0, hub_ind, :]
            print(dur, dur_ind, i, hub_ind)
        dur_hub[dur_ind] /= dur
        np.save(f'/Users/cschaefe/datasets/multispeaker_welt_bild/phon_hubert/{id}.npy', dur_hub)


