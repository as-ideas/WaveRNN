import unittest
import numpy as np
import torch

from duration_extraction.duration_extraction_pipe import DurationExtractionDataset


class TestDurationExtractionDataset(unittest.TestCase):

    def test_expand_attention_tensor(self) -> None:
        attention = torch.ones((4, 2))
        mel_mask = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1])

        attention_expanded = DurationExtractionDataset.expand_attention_tensor(attention, mel_mask)

        # the expected expanded attention tensor has zero rows for zero indices in mel mask (0, 2, 3, 6)
        expected = torch.ones((8, 2))
        expected[0, :] = 0
        expected[2, :] = 0
        expected[3, :] = 0
        expected[6, :] = 0

        np.testing.assert_allclose(expected.numpy(), attention_expanded.numpy())
