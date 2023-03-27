import unittest

from utils.dataset import DurationStats, DataFilter


class TestDatasetFilter(unittest.TestCase):

    def test_filter_happy_path(self) -> None:

        dur_stats = {
            'id_1': DurationStats(att_align_score=1., att_sharpness_score=1., max_consecutive_ones=1, max_duration=2),
            'id_2': DurationStats(att_align_score=0.5, att_sharpness_score=1., max_consecutive_ones=1, max_duration=2),
            'id_3': DurationStats(att_align_score=1., att_sharpness_score=0.5, max_consecutive_ones=1, max_duration=2),
            'id_4': DurationStats(att_align_score=1., att_sharpness_score=1., max_consecutive_ones=6, max_duration=2),
            'id_5': DurationStats(att_align_score=1., att_sharpness_score=1., max_consecutive_ones=1, max_duration=20),
        }

        dataset = [
            ('id_1', 1000),
            ('id_2', 1000),
            ('id_3', 1000),
            ('id_4', 1000),
            ('id_5', 5000),
        ]

        data_filter = DataFilter(duration_stats=dur_stats,
                                 min_attention_alignment=1.,
                                 min_attention_sharpness=1.,
                                 max_consecutive_duration_ones=1,
                                 max_duration=2)

        result = data_filter(dataset)

        self.assertEqual(['id_1'], [r for r, _ in result])

        data_filter = DataFilter(duration_stats=dur_stats,
                                 min_attention_alignment=0.,
                                 min_attention_sharpness=0.,
                                 max_consecutive_duration_ones=5,
                                 max_duration=10)

        result = data_filter(dataset)

        self.assertEqual(['id_1', 'id_2', 'id_3'], [r for r, _ in result])

