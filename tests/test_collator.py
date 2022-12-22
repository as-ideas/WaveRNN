import unittest
import numpy as np
import torch

from utils.dataset import TacoCollator, ForwardCollator


class TestDataset(unittest.TestCase):

    def test_collate_forward(self) -> None:
        items = [
            {
                'item_id': 0,
                'mel': np.full((2, 5), fill_value=1.),
                'x': np.full(2, fill_value=2.),
                'mel_len': 5,
                'x_len': 2,
                'dur': np.full(2, fill_value=3.),
                'pitch': np.full(2, fill_value=4.),
                'energy': np.full(2, fill_value=5.),
                'speaker_emb': np.full(1, fill_value=4.)
            },
            {
                'item_id': 1,
                'mel': np.full((2, 6), fill_value=1.),
                'x': np.full(3, fill_value=2.),
                'mel_len': 6,
                'x_len': 3,
                'dur': np.full(3, fill_value=3.),
                'pitch': np.full(3, fill_value=4.),
                'energy': np.full(3, fill_value=5.),
                'speaker_emb': np.full(1, fill_value=5.)
            }
        ]

        collator = ForwardCollator(taco_collator=TacoCollator(r=1))
        batch = collator(items)
        self.assertEqual(0, batch['item_id'][0])
        self.assertEqual(1, batch['item_id'][1])
        self.assertEqual((2, 7), batch['mel'][0].size())
        self.assertEqual((2, 7), batch['mel'][1].size())
        self.assertEqual([2., 2., 2., 2., 2., -11.5129*2, -11.5129*2], torch.sum(batch['mel'][0], dim=0).tolist())
        self.assertEqual([2., 2., 2., 2., 2., 2., -11.5129*2], torch.sum(batch['mel'][1], dim=0).tolist())
        self.assertEqual(2, batch['x_len'][0])
        self.assertEqual(3, batch['x_len'][1])
        self.assertEqual(5, batch['mel_len'][0])
        self.assertEqual(6, batch['mel_len'][1])
        self.assertEqual([2., 2., 0], batch['x'][0].tolist())
        self.assertEqual([2., 2., 2.], batch['x'][1].tolist())
        self.assertEqual([3., 3., 0], batch['dur'][0].tolist())
        self.assertEqual([3., 3., 3.], batch['dur'][1].tolist())
        self.assertEqual([4., 4., 0], batch['pitch'][0].tolist())
        self.assertEqual([4., 4., 4.], batch['pitch'][1].tolist())
        self.assertEqual([5., 5., 0], batch['energy'][0].tolist())
        self.assertEqual([5., 5., 5.], batch['energy'][1].tolist())
        self.assertEqual([4.], batch['speaker_emb'][0].tolist())
        self.assertEqual([5.], batch['speaker_emb'][1].tolist())

