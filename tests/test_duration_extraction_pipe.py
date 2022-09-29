import os
import shutil
import unittest
from pathlib import Path
from typing import Tuple
from unittest.mock import patch

import numpy as np
import torch

from duration_extraction.duration_extraction_pipe import DurationExtractionPipeline
from duration_extraction.duration_extractor import DurationExtractor
from models.tacotron import Tacotron
from utils.files import read_config, pickle_binary
from utils.paths import Paths


def new_diagonal_att(dims: Tuple[int, int, int]) -> torch.Tensor:
    """ We assume an almost square matrix of shape about (1, M, M) """
    att = torch.zeros(dims).float()
    for i in range(dims[1]):
        j = min(i, dims[2]-1)
        att[:, i, j] = 1
    return att


class MockTacotron(torch.nn.Module):

    def __call__(self, x: torch.Tensor, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return x, x, new_diagonal_att((1, mel.size(-1), x.size(-1)))


class TestDurationExtractionPipe(unittest.TestCase):

    def setUp(self) -> None:
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.resource_path = Path(test_path) / 'resources'
        self.config = read_config(self.resource_path / 'test_config.yaml')
        self.paths = Paths(data_path='tmp_resources/data_test', voc_id='voc_test_id', tts_id='tts_test_id')
        self.train_dataset = [('id_1', 5), ('id_2', 10), ('id_3', 15)]
        self.val_dataset = [('id_4', 6), ('id_5', 12)]
        pickle_binary(self.train_dataset, self.paths.data / 'train_dataset.pkl')
        pickle_binary(self.val_dataset, self.paths.data / 'val_dataset.pkl')
        self.text_dict = {id: 'a' * l for id, l in self.train_dataset + self.val_dataset}
        pickle_binary(self.text_dict, self.paths.data / 'text_dict.pkl')
        for id, mel_len in self.train_dataset + self.val_dataset:
            np.save(self.paths.mel / f'{id}.npy', np.ones((5, mel_len)), allow_pickle=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.paths.data.parent)

    @patch.object(Tacotron, '__call__', new_callable=MockTacotron)
    def test_extract_attentions_durations(self, mock_tacotron: Tacotron) -> None:

        duration_extractor = DurationExtractor(silence_threshold=-11.,
                                               silence_prob_shift=0.25)

        duration_extraction_pipe = DurationExtractionPipeline(paths=self.paths, config=self.config,
                                                              duration_extractor=duration_extractor)

        avg_att_score = duration_extraction_pipe.extract_attentions(model=mock_tacotron, batch_size=1)
        self.assertEqual(1., avg_att_score)
        att_files = list(self.paths.att_pred.glob('**/*.npy'))
        self.assertEqual(5, len(att_files))

        for item_id, mel_len in (self.train_dataset + self.val_dataset):
            att = np.load(self.paths.att_pred / f'{item_id}.npy')
            x = self.text_dict[item_id]
            expected_att_size = (mel_len, len(x))
            self.assertEqual(expected_att_size, att.shape)

        duration_extraction_pipe.extract_durations(num_workers=1, sampler_bin_size=1)

        dur_files = list(self.paths.alg.glob('**/*.npy'))
        self.assertEqual(5, len(dur_files))

        for dur_file in dur_files:
            dur = np.load(dur_file)
            expected = np.ones(len(dur)) # All ones because we created diagonal attentions
            np.testing.assert_allclose(expected, dur, rtol=1e-8)