import unittest
from typing import Tuple

import torch

from duration_extraction.duration_extractor import DurationExtractor


def new_diagonal_attention(dims: Tuple[int, int]) -> torch.Tensor:
    att = torch.zeros(dims).float()
    for i in range(dims[0]):
        att[i, i//2] = 1
    return att


class TestDurationExtractor(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_extract_happy_path(self) -> None:
        x = torch.tensor([15, 16, 10, 17, 18]).long()
        mel = torch.full((80, 10), fill_value=-10).float()
        attention = new_diagonal_attention((10, 5))
        duration_extractor = DurationExtractor(silence_threshold=-11.,
                                               silence_prob_shift=0.)
        durs, att_score = duration_extractor(x=x, mel=mel, attention=attention)
        expected = [2., 2., 2., 2., 2]
        self.assertEqual(expected, durs.tolist())

    def test_extract_with_silent_part(self) -> None:
        """ Test extraction for mel with silent part that suffers from fuzzy attention. """

        x = torch.tensor([15, 16, 10, 17, 18]).long()

        # Mock up mel that has silence at indices 4:6
        mel = torch.full((80, 10), fill_value=-10).float()
        mel[:, 4:6] = -11.51

        # Mock up simple diagonal attention which is fuzzy at mel indices 3:5, exactly where the model
        # should look at x[2], which is a silent token (token_index=10, which is a whitespace)
        attention = new_diagonal_attention((10, 5))
        attention[3:5, :] = 1./len(x)

        # duration extractor with no probability shift delivers larger durations after the pause (at index=3)
        duration_extractor = DurationExtractor(silence_threshold=-11.,
                                               silence_prob_shift=0.)
        durs, att_score = duration_extractor(x=x, mel=mel, attention=attention)
        expected = [2., 3., 1., 2., 2]
        self.assertEqual(expected, durs.tolist())

        # duration extractor with some probability shift delivers larger durations during the pause (at index=2)
        duration_extractor = DurationExtractor(silence_threshold=-11.,
                                               silence_prob_shift=0.25)
        durs, att_score = duration_extractor(x=x, mel=mel, attention=attention)
        expected = [2., 2., 2., 2., 2]
        self.assertEqual(expected, durs.tolist())
