import shutil
import unittest
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import torch

from duration_extraction.duration_extraction_pipe import DurationExtractionPipeline
from duration_extraction.duration_extractor import DurationExtractor
from models.forward_tacotron import ForwardTacotron
from models.tacotron import Tacotron
from utils.files import read_config, pickle_binary
from utils.paths import Paths


class MockTacotron(torch.nn.Module):

    def __call__(self, x: torch.Tensor, mel: torch.Tensor) -> Dict[str, Any]:
        return {}


def new_diagonal_att(dims: Tuple[int, int]) -> torch.Tensor:
    att = torch.zeros(dims).float()
    for i in range(dims[0]):
        att[i, i//2] = 1
    return att


class TestDurationExtractionPipe(unittest.TestCase):

    def setUp(self) -> None:
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.resource_path = Path(test_path) / 'resources'
        self.config = read_config(self.resource_path / 'test_config.yaml')
        self.paths = Paths(data_path='data_test', voc_id='voc_test_id', tts_id='tts_test_id')
        train_dataset = [('id_1', 5), ('id_2', 10)]
        val_dataset = [('id_3', 6), ('id_4', 12)]
        pickle_binary(train_dataset, self.paths.data / 'train_dataset.pkl')
        pickle_binary(val_dataset, self.paths.data / 'val_dataset.pkl')
        text_dict = {'id_1': 'a', 'id_2': 'aa', 'id_3': 'ab', 'id_4': 'abb'}
        pickle_binary(text_dict, self.paths.data / 'text_dict.pkl')


    #def tearDown(self) -> None:
    #    shutil.rmtree(self.paths.data)

    def test_extract_attentions(self) -> None:

        duration_extractor = DurationExtractor(silence_threshold=-11.,
                                               silence_prob_shift=0.25)

        duration_extraction_pipe = DurationExtractionPipeline(paths=self.paths, config=self.config,
                                                              duration_extractor=duration_extractor)

        model = MockTacotron()
        duration_extraction_pipe.extract_attentions(model=model, batch_size=1)

        print()