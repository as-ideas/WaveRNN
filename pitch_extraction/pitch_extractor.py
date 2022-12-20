from abc import ABC
import librosa
import numpy as np
try:
    import pyworld as pw
except ImportError as e:
    print('WARNING: Could not import pyworld! Please use pitch_extraction_method: librosa.')


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
                      frame_period=self.hop_length / self.sample_rate * 1000)