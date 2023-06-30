from pathlib import Path

import numpy as np
from typing import List
import tqdm

from utils.files import unpickle_binary

""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'

# Phonemes
_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
_diacrilics = 'ɚ˞ɫ'
_extra_phons = ['g', 'ɝ', '̃', '̍', '̥', '̩', '̯', '͡']  # some extra symbols that I found in from wiktionary ipa annotations

phonemes = list(
    _pad + _punctuation + _special + _vowels + _non_pulmonic_consonants
    + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics) + _extra_phons

phonemes_set = set(phonemes)
silent_phonemes_indices = [i for i, p in enumerate(phonemes) if p in _pad + _punctuation]


class Tokenizer:

    def __init__(self) -> None:
        self.symbol_to_id = {s: i for i, s in enumerate(phonemes)}
        self.id_to_symbol = {i: s for i, s in enumerate(phonemes)}

    def __call__(self, text: str) -> List[int]:
        return [self.symbol_to_id[t] for t in text if t in self.symbol_to_id]

    def decode(self, sequence: List[int]) -> str:
        text = [self.id_to_symbol[s] for s in sequence if s in self.id_to_symbol]
        return ''.join(text)


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
        all_durs = []
        for k, v in self.phon_dur.items():
            all_durs.extend(v)
        cat = np.array(all_durs)
        all_mean, all_std = np.mean(cat), np.std(cat)
        print('all mean std', self.speaker_id, all_mean, all_std)

        for k, v in self.phon_dur.items():
            if len(v) > 10:
                cat = np.array(v)
                mean, std = np.mean(cat), np.std(cat)
                if not 0 < mean < 20:
                    mean = all_mean
                if not 0 < std < 50:
                    std = all_std
            else:
                mean, std = all_mean, all_std
            self.phon_stat[k] = (mean, std)
        del self.phon_dur

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

        print('Finalize dur stats')
        for k, v in tqdm.tqdm(self.dur_stats.items(), total=len(self.dur_stats)):
            v.finalize()

    def normalize(self, speaker_id: str, dur: np.array, text: str) -> np.array:
        dur_stat = self.dur_stats[speaker_id]
        dur_norm = dur_stat.normalize(dur, text)
        return dur_norm

    def denormalize(self, speaker_id: str, dur: np.array, text: str) -> np.array:
        dur_stat = self.dur_stats[speaker_id]
        dur_denorm = dur_stat.denormalize(dur, text)
        return dur_denorm