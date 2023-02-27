import os
import unittest
from pathlib import Path

import librosa
import numpy as np

from utils.dsp import DSP
from utils.files import read_config


class TestDSP(unittest.TestCase):

    def setUp(self) -> None:
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.resource_path = Path(test_path) / 'resources'

    def test_melspectrogram(self) -> None:
        config = read_config(self.resource_path / 'test_config.yaml')
        dsp = DSP.from_config(config)
        file = librosa.util.example('brahms')
        y = dsp.load_wav(file)[:10000]
        mel = dsp.wav_to_mel(y)
        expected = np.load(str(self.resource_path / 'test_mel.npy'))
        np.testing.assert_allclose(expected, mel, rtol=1e-5)


