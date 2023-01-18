import os
import unittest
from pathlib import Path

import torch

from models.multi_forward_tacotron import MultiForwardTacotron
from utils.files import read_config


class TestMultiForwardTacotron(unittest.TestCase):

    def setUp(self) -> None:
        test_path = os.path.dirname(os.path.abspath(__file__))
        self.base_path = Path(test_path).parent
        config = read_config(self.base_path / 'configs/multispeaker.yaml')
        self.model = MultiForwardTacotron.from_config(config)

    def test_forward(self) -> None:
        """ Simple test of happy path forward pass, useful for debugging purposes. """

        batch = {
            'dur': torch.full((2, 10), fill_value=2).long(),
            'mel': torch.ones((2, 80, 20)).float(),
            'x': torch.ones((2, 10)).long(),
            'speaker_emb': torch.ones((2, 256)).float(),
            'mel_len': torch.full((2, ), fill_value=20).long(),
            'pitch': torch.ones((2, 10)).float(),
            'energy': torch.ones((2, 10)).float(),
            'pitch_cond': torch.ones((2, 10)).long(),
        }

        pred = self.model(batch)

        self.assertEqual({'mel', 'mel_post', 'dur', 'pitch', 'energy', 'pitch_cond'}, pred.keys())
        self.assertEqual((2, 80, 20), pred['mel_post'].size())
        self.assertEqual((2, 10), pred['dur'].size())
        self.assertEqual((2, 1, 10), pred['pitch'].size())
        self.assertEqual((2, 1, 10), pred['energy'].size())
        self.assertEqual((2, 10, 3), pred['pitch_cond'].size())

    def test_generate(self) -> None:
        gen = self.model.generate(x=torch.ones((1, 10)).long(), speaker_emb=torch.ones((1, 256)))
        self.assertEqual({'mel', 'mel_post', 'dur', 'pitch', 'energy', 'pitch_cond'}, gen.keys())
        self.assertEqual(80, gen['mel_post'].size(1))
        self.assertEqual((1, 10), gen['dur'].size())
        self.assertEqual((1, 1, 10), gen['pitch'].size())
        self.assertEqual((1, 1, 10), gen['energy'].size())
        self.assertEqual((1, 1, 10), gen['pitch_cond'].size())
