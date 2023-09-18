import unittest

import numpy as np
import torch

from trainer.common import new_guided_attention_matrix


class TestGuidedAttentionMatrix(unittest.TestCase):

    def test_happy_path(self):
        attention = torch.ones(1, 4, 3)  # Example input tensor with shape (batch_size, T, N)
        g = 1.0
        dia_mat = new_guided_attention_matrix(attention, g)

        expected = torch.tensor([
            [[1., 0.9460, 0.8007],
             [0.9692, 0.9965, 0.9169],
             [0.8825, 0.9862, 0.9862],
             [0.7548, 0.9169, 0.9965]]
        ], dtype=torch.float32)

        np.testing.assert_allclose(expected.numpy(), dia_mat.numpy(), atol=1e-4)