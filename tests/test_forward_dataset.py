import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from utils.dataset import ForwardDataset
from utils.text.tokenizer import Tokenizer


class TestForwardDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory(prefix='TestForwarDatasetTmp')

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_items(self) -> None:
        text_dict = {'0': 'a', '1': 'bc'}
        data_dir = Path(self.temp_dir.name + '/data')
        mel_dir = data_dir / 'mel'
        alg_dir = data_dir / 'alg'
        pitch_dir = data_dir / 'phon_pitch'
        energy_dir = data_dir / 'phon_energy'
        mel_dir.mkdir(parents=True)
        alg_dir.mkdir(parents=True)
        pitch_dir.mkdir(parents=True)
        energy_dir.mkdir(parents=True)

        mels = [np.full((2, 2), fill_value=1), np.full((2, 3), fill_value=2)]
        durs = [np.full(1, fill_value=2), np.full(2, fill_value=3)]
        pitches = [np.full(1, fill_value=5), np.full(2, fill_value=6)]
        energies = [np.full(1, fill_value=6), np.full(2, fill_value=7)]

        for i in range(2):
            np.save(str(mel_dir / f'{i}.npy'), mels[i])
            np.save(str(alg_dir / f'{i}.npy'), durs[i])
            np.save(str(pitch_dir / f'{i}.npy'), pitches[i])
            np.save(str(energy_dir / f'{i}.npy'), energies[i])

        dataset = ForwardDataset(path=data_dir,
                                 dataset_ids=['0', '1'],
                                 text_dict=text_dict,
                                 tokenizer=Tokenizer())

        data = [dataset[i] for i in range(len(dataset))]

        np.testing.assert_allclose(data[0]['mel'], mels[0], rtol=1e-10)
        np.testing.assert_allclose(data[1]['mel'], mels[1], rtol=1e-10)
        np.testing.assert_allclose(data[0]['dur'], durs[0], rtol=1e-10)
        np.testing.assert_allclose(data[1]['dur'], durs[1], rtol=1e-10)
        np.testing.assert_allclose(data[0]['pitch'], pitches[0], rtol=1e-10)
        np.testing.assert_allclose(data[1]['pitch'], pitches[1], rtol=1e-10)
        np.testing.assert_allclose(data[0]['energy'], energies[0], rtol=1e-10)
        np.testing.assert_allclose(data[1]['energy'], energies[1], rtol=1e-10)

        self.assertEqual(1, data[0]['x_len'])
        self.assertEqual(2, data[1]['x_len'])
        self.assertEqual('0', data[0]['item_id'])
        self.assertEqual('1', data[1]['item_id'])
        self.assertEqual(2, data[0]['mel_len'])
        self.assertEqual(3, data[1]['mel_len'])
