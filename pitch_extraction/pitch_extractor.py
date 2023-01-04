from abc import ABC
from enum import Enum
from typing import Dict, Any

import librosa
import numpy as np
try:
    import pyworld as pw
except ImportError as e:
    print('WARNING: Could not import pyworld! Please use pitch_extraction_method: librosa.')


class PitchExtractionMethod(Enum):
    LIBROSA = 'librosa'
    PYWORLD = 'pyworld'


class PitchExtractor(ABC):

    def __call__(self, wav: np.array) -> np.array:
        raise NotImplementedError()


class LibrosaPitchExtractor(PitchExtractor):

    def __init__(self,
                 fmin: int,
                 fmax: int,
                 sample_rate: int,
                 frame_length: int,
                 hop_length: int) -> None:

        self.fmin = fmin
        self.fmax= fmax
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length

    def __call__(self, wav: np.array) -> np.array:
        pitch, _, _ = librosa.pyin(wav,
                                   fmin=self.fmin,
                                   fmax=self.fmax,
                                   sr=self.sample_rate,
                                   frame_length=self.frame_length,
                                   hop_length=self.hop_length)
        np.nan_to_num(pitch, copy=False, nan=0.)
        return pitch


class PyworldPitchExtractor(PitchExtractor):

    def __init__(self,
                 sample_rate: int,
                 hop_length: int) -> None:

        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def __call__(self, wav: np.array) -> np.array:
        return pw.dio(wav.astype(np.float64), self.sample_rate,
                      frame_period=self.hop_length / self.sample_rate * 1000)[0]


def new_pitch_extractor_from_config(config: Dict[str, Any]) -> PitchExtractor:
    preproc_config = config['preprocessing']
    pitch_extractor_type = preproc_config['pitch_extractor']
    if pitch_extractor_type == 'librosa':
        pitch_extractor = LibrosaPitchExtractor(fmin=preproc_config['pitch_min_freq'],
                                                fmax=preproc_config['pitch_max_freq'],
                                                frame_length=preproc_config['pitch_frame_length'],
                                                sample_rate=config['dsp']['sample_rate'],
                                                hop_length=config['dsp']['hop_length'])
    elif pitch_extractor_type == 'pyworld':
        pitch_extractor = PyworldPitchExtractor(hop_length=config['dsp']['hop_length'],
                                                sample_rate=config['dsp']['sample_rate'])
    else:
        raise ValueError(f'Invalid pitch extractor type: {pitch_extractor_type}, choices: [librosa, pyworld].')
    return pitch_extractor
