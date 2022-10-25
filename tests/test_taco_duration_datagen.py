import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from utils.dataset import get_taco_duration_extraction_generator
from utils.files import read_config, pickle_binary
from utils.paths import Paths


class TestTacoDurationDatagen(unittest.TestCase):

    def setUp(self) -> None:
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.resource_path = Path(test_path) / 'resources'
        self.config = read_config(self.resource_path / 'test_config.yaml')
        self.temp_dir = TemporaryDirectory(prefix='forwardtaco_data_test_temp')
        self.paths = Paths(data_path=self.temp_dir.name + '/data', voc_id='voc_test_id', tts_id='tts_test_id')
        self.dataset = [('id_1', 2), ('id_2', 2), ('id_3', 3), ('id_4', 4), ('id_5', 4), ('id_6', 4)]
        self.text_dict = {file_id: 'a' * length for file_id, length in self.dataset}
        pickle_binary(self.text_dict, self.paths.data / 'text_dict.pkl')
        for id, mel_len in self.dataset:
            np.save(self.paths.mel / f'{id}.npy', np.ones((5, mel_len)), allow_pickle=False)
            np.save(self.paths.speaker_emb / f'{id}.npy', np.ones(256), allow_pickle=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_items(self) -> None:
        size, generator = get_taco_duration_extraction_generator(self.paths.data,
                                                                 dataset=self.dataset,
                                                                 max_batch_size=2)

        expected_xlen_batches = [[2, 2], [3], [4, 4], [4]]
        self.assertEqual(len(expected_xlen_batches), size)

        batches = [d for d in generator]
        batches.sort(key=lambda b: b['x_len'][0])

        actual_x_len_batches = [b['x_len'].tolist() for b in batches]
        self.assertEqual(expected_xlen_batches, actual_x_len_batches)

