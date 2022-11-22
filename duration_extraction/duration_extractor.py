from typing import Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from utils.text.symbols import silent_phonemes_indices


class DurationExtractor:

    def __init__(self,
                 silence_threshold: float,
                 silence_prob_shift: float) -> None:
        """
        :param silence_threshold: Mel spec threshold below which the voice is considered silent.
        :param silence_prob_shift: Attention probability that is added to silent phonemes in unvoiced parts.
        """
        self.silence_prob_shift = silence_prob_shift
        self.silence_threshold = silence_threshold

    def __call__(self,
                 x: torch.Tensor,
                 mel: torch.Tensor,
                 attention: torch.Tensor) -> Tuple[torch.tensor, float]:
        """
        Extracts durations from the attention matrix by finding the shortest monotonic path from
        top left to bottom right.

        :param x: Tokenized sequence.
        :param mel: Mel spec.
        :param attention: Attention matrix with shape (mel_len, x_len).
        :return: Tuple, where the first entry is the durations and the second entry is the average attention probability.
        """
        attention = attention[...]
        mel_len = mel.shape[-1]

        # We add a little probability to silent phonemes within unvoiced parts of the spec where the tacotron attention
        # is usually very unreliable. As a result we get more accurate (larger) durations for unvoiced parts and
        # avoid 'leakage' of durations into surrounding word phonemes.
        sil_mask = mel.mean(dim=0) < self.silence_threshold
        sil_mel_inds = sil_mask.nonzero().squeeze()
        sil_mel_inds = list(sil_mel_inds) if len(sil_mel_inds.size()) > 0 else []

        sil_phon_inds = torch.tensor(silent_phonemes_indices)
        for i in sil_mel_inds:
            sil_tok_inds = torch.isin(x, sil_phon_inds)
            att_shift = sil_tok_inds.float() * self.silence_prob_shift * 2 - self.silence_prob_shift
            attention[i, :] = attention[i, :] + att_shift

        attention = torch.clamp(attention, min=0., max=1.)
        path_probs = 1. - attention[:mel_len, :]
        adj_matrix = self._to_adj_matrix(path_probs)
        dist_matrix, predecessors = dijkstra(csgraph=adj_matrix, directed=True,
                                             indices=0, return_predecessors=True)
        path = []
        pr_index = predecessors[-1]
        while pr_index != 0:
            path.append(pr_index)
            pr_index = predecessors[pr_index]
        path.reverse()

        # append first and last node
        path = [0] + path + [dist_matrix.size-1]
        cols = path_probs.shape[1]
        mel_text = {}
        durations = torch.zeros(x.shape[0])

        att_scores = []

        # collect indices (mel, text) along the path
        for node_index in path:
            i, j = self._from_node_index(node_index, cols)
            mel_text[i] = j
            if not sil_mask[i]:
                att_scores.append(float(attention[i, j]))

        for j in mel_text.values():
            durations[j] += 1

        att_score = sum(att_scores) / len(att_scores)

        return durations, att_score

    @staticmethod
    def _to_node_index(i: int, j: int, cols: int) -> int:
        return cols * i + j

    @staticmethod
    def _from_node_index(node_index: int, cols: int) -> Tuple[int, int]:
        return node_index // cols, node_index % cols

    @staticmethod
    def _to_adj_matrix(mat: np.array) -> csr_matrix:
        rows = mat.shape[0]
        cols = mat.shape[1]

        row_ind = []
        col_ind = []
        data = []

        for i in range(rows):
            for j in range(cols):

                node = DurationExtractor._to_node_index(i, j, cols)

                if j < cols - 1:
                    right_node = DurationExtractor._to_node_index(i, j + 1, cols)
                    weight_right = mat[i, j + 1]
                    row_ind.append(node)
                    col_ind.append(right_node)
                    data.append(weight_right)

                if i < rows - 1 and j < cols:
                    bottom_node = DurationExtractor._to_node_index(i + 1, j, cols)
                    weight_bottom = mat[i + 1, j]
                    row_ind.append(node)
                    col_ind.append(bottom_node)
                    data.append(weight_bottom)

                if i < rows - 1 and j < cols - 1:
                    bottom_right_node = DurationExtractor._to_node_index(i + 1, j + 1, cols)
                    weight_bottom_right = mat[i + 1, j + 1]
                    row_ind.append(node)
                    col_ind.append(bottom_right_node)
                    data.append(weight_bottom_right)

        adj_mat = coo_matrix((data, (row_ind, col_ind)), shape=(rows * cols, rows * cols))
        return adj_mat.tocsr()
