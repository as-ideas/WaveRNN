from pathlib import Path
import torch
import numpy as np
import tqdm

from utils.files import unpickle_binary

device = torch.device('cpu')


hubert_files = list(Path('/Users/cschaefe/datasets/multispeaker_welt_bild/hubert').glob('**/*.pt'))


text_dict = unpickle_binary('/Users/cschaefe/datasets/multispeaker_welt_bild/text_dict.pkl')

for hub in tqdm.tqdm(hubert_files, total=len(hubert_files)):
    hubert = torch.load(hub)
    id = hub.stem
    durs = np.load(f'/Users/cschaefe/datasets/multispeaker_welt_bild/alg/{id}.npy')
    hub_fact = hubert.size(1) / durs.sum()
    dur_hub = torch.zeros((int(durs.shape[0]), 1024))
    dur_ind = 0
    total_dur_ind = 0
    for dur_ind, dur in enumerate(durs):
        for i in range(dur):
            hub_ind = min(int(hub_fact * total_dur_ind), hubert.size(1)-1)
            dur_hub[dur_ind] += hubert[0, hub_ind, :]
            total_dur_ind += 1
            #print(dur_ind, dur, total_dur_ind, hub_ind, hubert.size())
        dur_hub[dur_ind] /= dur

    dur_hub = dur_hub.transpose(0, 1).cpu().numpy()
    #print('npy shape: ', dur_hub.shape)

    np.save(f'/Users/cschaefe/datasets/multispeaker_welt_bild/phon_hubert/{id}.npy', dur_hub)


