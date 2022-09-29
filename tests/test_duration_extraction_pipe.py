import shutil
import unittest
import os
from unittest.mock import patch

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

import torch

from duration_extraction.duration_extraction_pipe import DurationExtractionPipeline
from duration_extraction.duration_extractor import DurationExtractor
from models.forward_tacotron import ForwardTacotron
from models.tacotron import Tacotron
from utils.files import read_config, pickle_binary
from utils.paths import Paths


def new_diagonal_att(dims: Tuple[int, int, int]) -> torch.Tensor:
    att = torch.zeros(dims).float()
    for i in range(dims[1]):
        j = dims[2] * i // dims[1]
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
        self.paths = Paths(data_path='data_test', voc_id='voc_test_id', tts_id='tts_test_id')
        train_dataset = [('id_1', 5), ('id_2', 10), ('id_3', 15)]
        val_dataset = [('id_4', 6), ('id_5', 12)]
        pickle_binary(train_dataset, self.paths.data / 'train_dataset.pkl')
        pickle_binary(val_dataset, self.paths.data / 'val_dataset.pkl')
        text_dict = {id: 'a' * l for id, l in train_dataset + val_dataset}
        pickle_binary(text_dict, self.paths.data / 'text_dict.pkl')
        for id, mel_len in train_dataset + val_dataset:
            np.save(self.paths.mel / f'{id}.npy', np.ones((5, mel_len)), allow_pickle=False)


    #def tearDown(self) -> None:
    #    shutil.rmtree(self.paths.data)


    @patch.object(Tacotron, '__call__', new_callable=MockTacotron)
    def test_extract_attentions(self, mock_tacotron: Tacotron) -> None:

        duration_extractor = DurationExtractor(silence_threshold=-11.,
                                               silence_prob_shift=0.25)

        duration_extraction_pipe = DurationExtractionPipeline(paths=self.paths, config=self.config,
                                                              duration_extractor=duration_extractor)

        duration_extraction_pipe.extract_attentions(model=mock_tacotron, batch_size=1)

        print()