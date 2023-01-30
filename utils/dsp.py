import math
import struct
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import librosa
import webrtcvad
import soundfile as sf
from scipy.ndimage import binary_dilation

from utils.files import unpickle_binary


class DSP:

    def __init__(self,
                 num_mels: int,
                 sample_rate: int,
                 hop_length: int,
                 win_length: int,
                 n_fft: int,
                 fmin: float,
                 fmax: float,
                 peak_norm: bool,
                 trim_start_end_silence: bool,
                 trim_silence_top_db:  int,
                 pitch_max_freq: int,
                 trim_long_silences: bool,
                 vad_sample_rate: int,
                 vad_window_length: float,
                 vad_moving_average_width: float,
                 vad_max_silence_length: int,
                 bits: int,
                 mu_law: bool,
                 voc_mode: str,
                 ) -> None:

        self.n_mels = num_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.scaler = unpickle_binary('assets/bild_scaler.pkl')

        self.should_peak_norm = peak_norm
        self.should_trim_start_end_silence = trim_start_end_silence
        self.should_trim_long_silences = trim_long_silences
        self.trim_silence_top_db = trim_silence_top_db
        self.pitch_max_freq = pitch_max_freq

        self.vad_sample_rate = vad_sample_rate
        self.vad_window_length = vad_window_length
        self.vad_moving_average_width = vad_moving_average_width
        self.vad_max_silence_length = vad_max_silence_length

        self.bits = bits
        self.mu_law = mu_law
        self.voc_mode = voc_mode

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DSP':
        return DSP(**config['dsp'])

    def load_wav(self, path: Union[str, Path]) -> np.array:
        wav, _ = librosa.load(str(path), sr=self.sample_rate)
        return wav

    def save_wav(self, wav: np.array, path: Union[str, Path]) -> None:
        wav = wav.astype(np.float32)
        sf.write(str(path), wav, samplerate=self.sample_rate)

    def wav_to_mel(self, y: np.array, normalize=True) -> np.array:
        spec = librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length)
        spec = np.abs(spec)
        mel = librosa.feature.melspectrogram(
            S=spec,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax)
        if normalize:
            mel = self.normalize(mel)
        return mel

    def griffinlim(self, mel: np.array, n_iter=32) -> np.array:
        mel = self.denormalize(mel)
        S = librosa.feature.inverse.mel_to_stft(
            mel,
            power=1,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            fmin=self.fmin,
            fmax=self.fmax)
        wav = librosa.core.griffinlim(
            S,
            n_iter=n_iter,
            hop_length=self.hop_length,
            win_length=self.win_length)
        return wav

    def normalize(self, mel: np.array) -> np.array:
        mel = np.clip(mel, a_min=1.e-5, a_max=None)
        return np.log(mel)

    def denormalize(self, mel: np.array) -> np.array:
        mel = (mel * np.expand_dims(self.scaler.scale_, axis=1) + np.expand_dims(self.scaler.mean_, axis=1))
        return np.exp(mel)

    def trim_silence(self, wav: np.array) -> np.array:
        return librosa.effects.trim(wav, top_db=self.trim_silence_top_db, frame_length=2048, hop_length=512)[0]

    # borrowed from https://github.com/resemble-ai/Resemblyzer/blob/master/resemblyzer/audio.py
    def trim_long_silences(self, wav: np.array) -> np.array:
        int16_max = (2 ** 15) - 1
        samples_per_window = (self.vad_window_length * self.vad_sample_rate) // 1000
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                             sample_rate=self.vad_sample_rate))
        voice_flags = np.array(voice_flags)
        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width
        audio_mask = moving_average(voice_flags, self.vad_moving_average_width)
        audio_mask = np.round(audio_mask).astype(np.bool)
        audio_mask[:] = binary_dilation(audio_mask[:], np.ones(self.vad_max_silence_length + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)
        return wav[audio_mask]

    @staticmethod
    def label_2_float(x: np.array, bits: float) -> np.array:
        return 2 * x / (2**bits - 1.) - 1.

    @staticmethod
    def float_2_label(x: np.array, bits: float) -> np.array:
        assert abs(x).max() <= 1.0
        x = (x + 1.) * (2**bits - 1) / 2
        return x.clip(0, 2**bits - 1)

    @staticmethod
    def encode_mu_law(x: np.array, mu: float) -> np.array:
        mu = mu - 1
        fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
        return np.floor((fx + 1) / 2 * mu + 0.5)

    @staticmethod
    def decode_mu_law(y: np.array, mu: float, from_labels=True) -> np.array:
        if from_labels:
            y = DSP.label_2_float(y, math.log2(mu))
        mu = mu - 1
        x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
        return x

