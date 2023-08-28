import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, Dict
from unittest.mock import patch

import numpy as np
import torch

from duration_extraction.duration_extraction_pipe import DurationExtractionPipeline
from duration_extraction.duration_extractor import DurationExtractor
from models.tacotron import Tacotron
from utils.files import read_config, pickle_binary
from utils.paths import Paths


def new_diagonal_attention(dims: Tuple[int, int, int]) -> torch.Tensor:
    """ Returns perfect diagonal attention matrix, assuming that the dimensions are almost square (1, M, M) """
    att = torch.zeros(dims).float()
    for i in range(dims[1]):
        j = min(i, dims[2]-1)
        att[:, i, j] = 1
    return att


class MockTacotron(torch.nn.Module):

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ We just use the mock model to get the returned diagonal attention matrix. """
        mel = batch['mel']
        x = batch['x']
        return {'att': new_diagonal_attention((1, mel.size(-1), x.size(-1)))}


class TestDurationExtractionPipe(unittest.TestCase):

    def setUp(self) -> None:
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.resource_path = Path(test_path) / 'resources'
        self.config = read_config(self.resource_path / 'test_config.yaml')
        self.temp_dir = TemporaryDirectory(prefix='TestDurationExtractionPipeTmp')
        self.paths = Paths(data_path=self.temp_dir.name + '/data', tts_id='tts_test_id')
        self.train_dataset = [('id_1', 5), ('id_2', 10), ('id_3', 15)]
        self.val_dataset = [('id_4', 6), ('id_5', 12)]
        pickle_binary(self.train_dataset, self.paths.train_dataset)
        pickle_binary(self.val_dataset, self.paths.val_dataset)
        self.text_dict = {file_id: 'a' * length for file_id, length in self.train_dataset + self.val_dataset}
        self.speaker_dict = {file_id: 'default_speaker' for file_id, _ in self.train_dataset + self.val_dataset}
        pickle_binary(self.text_dict, self.paths.text_dict)
        pickle_binary(self.speaker_dict, self.paths.speaker_dict)
        for id, mel_len in self.train_dataset + self.val_dataset:
            np.save(self.paths.mel / f'{id}.npy', np.ones((5, mel_len)), allow_pickle=False)
            np.save(self.paths.speaker_emb / f'{id}.npy', np.ones(1), allow_pickle=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @patch.object(Tacotron, '__call__', new_callable=MockTacotron)
    def test_extract_attentions_durations(self, mock_tacotron: Tacotron) -> None:

        duration_extractor = DurationExtractor(silence_threshold=-11.,
                                               silence_prob_shift=0.25)

        duration_extraction_pipe = DurationExtractionPipeline(paths=self.paths, config=self.config,
                                                              duration_extractor=duration_extractor)

        avg_att_score = duration_extraction_pipe.extract_attentions(model=mock_tacotron, max_batch_size=1)
        self.assertEqual(1., avg_att_score)
        att_files = list(self.paths.att_pred.glob('**/*.npy'))
        self.assertEqual(5, len(att_files))

        for item_id, mel_len in (self.train_dataset + self.val_dataset):
            att = np.load(self.paths.att_pred / f'{item_id}.npy')
            x = self.text_dict[item_id]
            expected_att_size = (mel_len, len(x))
            self.assertEqual(expected_att_size, att.shape)

        dur_stats = duration_extraction_pipe.extract_durations(num_workers=1, sampler_bin_size=1)

        for file_id, dur_stat in dur_stats.items():
            self.assertEqual(dur_stat.att_align_score, 1.)
            self.assertEqual(dur_stat.att_sharpness_score, 1.)
            self.assertTrue(3 <= dur_stat.max_consecutive_ones <= 15)
            self.assertEqual(dur_stat.max_duration, 1)

        dur_files = list(self.paths.alg.glob('**/*.npy'))
        self.assertEqual(5, len(dur_files))

        for dur_file in dur_files:
            dur = np.load(dur_file)
            # We expect durations of one due to the diagonal attention.
            expected = np.ones(len(dur))
            np.testing.assert_allclose(expected, dur, rtol=1e-8)
