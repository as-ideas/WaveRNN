import pandas as pd
from pathlib import Path
import numpy as np
import tqdm

from utils.files import unpickle_binary
from utils.text.symbols import phonemes, phonemes_set
from utils.text.tokenizer import Tokenizer


class DurStat:

    def __init__(self, speaker_id: str):
        self.speaker_id = speaker_id
        self.tokenizer = Tokenizer()
        self.phon_dur = {}
        self.phon_stat = {}
        for p in phonemes:
            self.phon_dur[p] = []

    def update(self, text: str, durs: np.array):
        text = ''.join([t for t in text if t in phonemes_set])
        for p, d in zip(text, durs):
            self.phon_dur[p].append(d)

    def finalize(self):
        for k, v in self.phon_dur.items():
            if len(v) > 5:
                cat = np.array(v)
                mean, std = np.mean(cat), np.std(cat)
            else:
                mean, std = 4, 1
            self.phon_stat[k] = (mean, std)
            print(self.speaker_id, k, mean, std)

    def normalize(self, dur: np.array, text: str) -> np.array:
        text = ''.join([t for t in text if t in phonemes_set])
        dur_norm = np.ones(dur.shape)
        for i, p in enumerate(text):
            mean, std = self.phon_stat[p]
            d = (dur[i] - mean) / std
            dur_norm[i] = d

        return dur_norm

    def denormalize(self, dur: np.array, text: str) -> np.array:
        text = ''.join([t for t in text if t in phonemes_set])
        dur_denorm = np.ones(dur.shape)
        for i, p in enumerate(text):
            mean, std = self.phon_stat[p]
            d = dur[i] * std + mean
            dur_denorm[i] = d

        return dur_denorm


class DurationNormalizer:

    def __init__(self, speaker_path: Path, dur_path: Path, text_path: Path):
        self.dur_path = dur_path
        self.text_dict = unpickle_binary(text_path)
        self.speaker_dict = unpickle_binary(speaker_path)

        print('Collect speakers')
        dur_files = list(dur_path.glob('**/*.npy'))
        speakers = set()
        for dur_file in tqdm.tqdm(dur_files, total=len(dur_files)):
            id = dur_file.stem
            speaker_id = self.speaker_dict[id]
            speakers.add(speaker_id)

        print('Calc dur stats')
        self.dur_stats = {speaker: DurStat(speaker) for speaker in speakers}
        for dur_file in tqdm.tqdm(dur_files, total=len(dur_files)):
            id = dur_file.stem
            speaker_id = self.speaker_dict[id]
            dur = np.load(str(self.dur_path / f'{id}.npy'))
            text = self.text_dict[id]
            self.dur_stats[speaker_id].update(text, dur)

        for k, v in self.dur_stats.items():
            v.finalize()

    def normalize(self, speaker_id: str, dur: np.array, text: str) -> np.array:
        dur_stat = self.dur_stats[speaker_id]
        dur_norm = dur_stat.normalize(dur, text)
        return dur_norm

    def denormalize(self, speaker_id: str, dur: np.array, text: str) -> np.array:
        dur_stat = self.dur_stats[speaker_id]
        dur_denorm = dur_stat.denormalize(dur, text)
        return dur_denorm


if __name__ == '__main__':

    dur_path = Path('/Users/cschaefe/datasets/multispeaker_welt_bild/alg')
    text_path = Path('/Users/cschaefe/datasets/multispeaker_welt_bild/text_dict.pkl')
    speaker_path = Path('/Users/cschaefe/datasets/multispeaker_welt_bild/speaker_dict.pkl')
    text_dict = unpickle_binary(text_path)
    speaker_dict = unpickle_binary(speaker_path)

    duration_normalizer = DurationNormalizer(speaker_path, dur_path, text_path)

    for id, text in text_dict.items():

        print(id)
        durs = np.load(str(dur_path / f'{id}.npy'))
        speaker_id =speaker_dict[id]
        dur_norm = duration_normalizer.normalize(speaker_id, durs, text)
        dur_denorm = duration_normalizer.denormalize(speaker_id, dur_norm, text)
        print('orig:', durs)
        print('norm:', dur_norm)
        print('denorm:', dur_denorm)
