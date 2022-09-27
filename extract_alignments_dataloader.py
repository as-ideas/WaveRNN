from pathlib import Path
import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Union

from utils.dsp import DSP
from utils.duration_extractor import DurationExtractor
from utils.files import read_config, unpickle_binary
from utils.metrics import attention_score
from utils.paths import Paths
from utils.text.tokenizer import Tokenizer



class DurationDataset(Dataset):

    def __init__(self,
                 duration_extractor: DurationExtractor,
                 paths: Paths,
                 dataset_ids: List[str],
                 text_dict: Dict[str, str],
                 tokenizer: Tokenizer):
        self.metadata = dataset_ids
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.text_dict = text_dict
        self.duration_extractor = duration_extractor
        self.paths = paths

    def __getitem__(self, index: int) -> tuple:

        item_id = self.metadata[index]
        x = self.text_dict[item_id]
        x = self.tokenizer(x)
        mel = np.load(self.paths.mel / f'{item_id}.npy')
        mel = torch.from_numpy(mel)
        x = torch.tensor(x)
        att_npy = np.load(str(self.paths.att_pred / f'{item_id}.npy'))
        att = torch.from_numpy(att_npy)
        mel_len = mel.shape[-1]
        mel_len = torch.tensor(mel_len).unsqueeze(0)
        align_score, _ = attention_score(att.unsqueeze(0), mel_len, r=1)
        durs, att_score = self.duration_extractor(x=x, mel=mel, att=att)
        #durs_npy = durs.cpu().numpy()
        print(item_id, att_score, durs)
        return item_id, att_score, align_score, durs.cpu()

    def __len__(self):
        return len(self.metadata)



if __name__ == '__main__':
    config = read_config('config.yaml')
    dsp = DSP.from_config(config)
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])

    text_dict = unpickle_binary(paths.data / 'text_dict.pkl')
    train_ids = list(text_dict.keys())
    train_ids = [t for t in train_ids if (paths.att_pred / f'{t}.npy').is_file()]

    duration_extractor = DurationExtractor(silence_prob_shift=config['preprocessing']['silence_prob_shift'],
                                           silence_threshold=config['preprocessing']['silence_threshold'])
    train_dataset = DurationDataset(
        duration_extractor=duration_extractor,
        paths=paths, dataset_ids=train_ids,
        text_dict=text_dict, tokenizer=Tokenizer())

    train_set = DataLoader(train_dataset, batch_size=1, num_workers=0,
                           pin_memory=True, collate_fn=None)
    for item_id, att_score, align_score, durs in tqdm.tqdm(train_set, total=len(train_set)):
        print(item_id, att_score, align_score, durs)
        np.save(paths.data / f'alg_extr/{item_id}.npy', durs.numpy(), allow_pickle=False)

