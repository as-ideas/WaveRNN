import struct
from pathlib import Path
from typing import Dict, Any, Union, List
import numpy as np
import librosa
import torch
import webrtcvad
from scipy.ndimage import binary_dilation
import torchaudio
import torchaudio.transforms as transforms

from utils.dataset import tensor_to_ndarray, ndarray_to_tensor


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
                 trim_long_silences: bool,
                 vad_sample_rate: int,
                 vad_window_length: float,
                 vad_moving_average_width: float,
                 vad_max_silence_length: int,
                 **kwargs,  # for backward compatibility
                 ) -> None:

        self.n_mels = num_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax

        self.should_peak_norm = peak_norm
        self.should_trim_start_end_silence = trim_start_end_silence
        self.should_trim_long_silences = trim_long_silences
        self.trim_silence_top_db = trim_silence_top_db

        self.vad_sample_rate = vad_sample_rate
        self.vad_window_length = vad_window_length
        self.vad_moving_average_width = vad_moving_average_width
        self.vad_max_silence_length = vad_max_silence_length

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # init transformation
        self.mel_transform = self._init_mel_transform()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DSP':
        """Initialize from configuration object"""
        return DSP(**config['dsp'])

    def _init_mel_transform(self):
        """Initialize mel transformation"""
        mel_transform = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=1,
            norm="slaney",
            n_mels=self.n_mels,
            mel_scale="slaney",
            f_min=self.fmin,
            f_max=self.fmax,
        ).to(self.device)

        return mel_transform

    def load_wav(self, path: Union[str, Path], mono: bool = True) -> torch.Tensor:
        """Load audio file into a tensor"""
        effects = []
        metadata = torchaudio.info(path)

        # merge channels if source is multichannel
        if mono and metadata.num_channels > 1:
            effects.extend([
                ["remix", "-"] # convert to mono
            ])

        # resample if source sample rate is different from desired sample rate
        if metadata.sample_rate != self.sample_rate:
            effects.extend([
                ["rate", f'{self.sample_rate}'],
            ])

        waveform, _ = torchaudio.sox_effects.apply_effects_file(path, effects=effects)
        return waveform

    def save_wav(self, waveform: torch.Tensor, path: Union[str, Path]) -> None:
        """Save waveform to file"""
        if not isinstance(waveform,torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(waveform)}")
        torchaudio.save(filepath=path, src=waveform, sample_rate=self.sample_rate)

    def adjust_volume(self, waveform: torch.Tensor, target_dbfs: int = -30) -> torch.Tensor:
        """Adjust volume of the waveform"""
        volume_transform = transforms.Vol(gain=target_dbfs, gain_type='db').to(self.device)
        return volume_transform(waveform)

    def adjust_volume_batched(self, data: List[torch.Tensor], target_dbfs: int = -30) -> List[torch.Tensor]:
        """Adjust volume of the waveforms in the batch"""
        lengths = [tensor.size(1) for tensor in data]
        padded_batch = [torch.nn.functional.pad(x, (0, max(lengths) - x.size(1))) for x in data]
        stacked_tensor = torch.stack(padded_batch, dim=0)
        processed_batch = self.adjust_volume(stacked_tensor, target_dbfs=target_dbfs)
        result = [processed_waveform[:, :lengths[index]] for index, processed_waveform in enumerate(processed_batch)]
        return result

    def waveform_to_mel_batched(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Convert waveform to mel spectrogram for the batch of waveforms"""
        lengths = [tensor.size(1) for tensor in batch]
        expected_mel_lengths = [x // self.hop_length + 1 for x in lengths]
        padded_batch = [torch.nn.functional.pad(x, (0, max(lengths) - x.size(1))) for x in batch]
        batch_tensor = torch.stack(padded_batch, dim=0).to(self.device)
        mels = self.waveform_to_mel(batch_tensor)
        list_of_mels = [mel[:, :, :expected_mel_lengths[index]] for index, mel in enumerate(mels)]
        return list_of_mels

    def waveform_to_mel(self, waveform: torch.Tensor, normalized: bool = True) -> torch.Tensor:
        """Convert waveform to mel spectrogram"""
        mel_spec = self.mel_transform(waveform)
        if normalized:
            mel_spec = self.normalize(mel_spec)
        return mel_spec

    def griffinlim(self, mel: np.array, n_iter: int = 32) -> np.array:
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

    @staticmethod
    def normalize(mel: torch.Tensor) -> torch.Tensor:
        """Normalize mel spectrogram"""
        mel = torch.clip(mel, min=1.e-5, max=None)
        return torch.log(mel)

    @staticmethod
    def denormalize(mel: np.ndarray) -> np.ndarray:
        """Denormalize mel spectrogram"""
        return np.exp(mel)

    def trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Trim silence from the waveform"""
        waveform = tensor_to_ndarray(waveform)
        trimmed_waveform = librosa.effects.trim(waveform,
                                                top_db=self.trim_silence_top_db,
                                                frame_length=self.win_length,
                                                hop_length=self.hop_length)
        return ndarray_to_tensor(trimmed_waveform[0])

    # borrowed from https://github.com/resemble-ai/Resemblyzer/blob/master/resemblyzer/audio.py
    def trim_long_silences(self, wav: torch.Tensor) -> torch.Tensor:
        wav = tensor_to_ndarray(wav)
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
        return ndarray_to_tensor(wav[audio_mask])
