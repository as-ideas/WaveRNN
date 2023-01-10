import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.text.recipes import read_ljspeech_format, read_pandas_format


# These tests have been written by ChatGPT and slightly adjusted


class TestRecipes(unittest.TestCase):

    def test_read_ljspeech_format_multi(self):
        test_data = "file_id_1|speaker1|Text1\nfile_id_2|speaker2|Text2\nfile_id_3|Text3"

        with tempfile.NamedTemporaryFile(mode='w+') as f:
            f.write(test_data)
            f.seek(0)
            path = Path(f.name)

            text_dict, speaker_dict = read_ljspeech_format(path, multispeaker=True)

            self.assertEqual({'file_id_1': 'Text1', 'file_id_2': 'Text2', 'file_id_3': 'Text3'}, text_dict)
            self.assertEqual({'file_id_1': 'speaker1', 'file_id_2': 'speaker2',
                              'file_id_3': 'default_speaker'}, speaker_dict)

    def test_read_ljspeech_format(self):
        test_data = "file_id_1|Text1\nfile_id_2|Text2"
        with tempfile.NamedTemporaryFile(mode='w+') as f:
            f.write(test_data)
            f.seek(0)
            path = Path(f.name)

            text_dict, speaker_dict = read_ljspeech_format(path, multispeaker=False)

            self.assertEqual({'file_id_1': 'Text1', 'file_id_2': 'Text2'}, text_dict)
            self.assertEqual({'file_id_1': 'default_speaker', 'file_id_2': 'default_speaker'}, speaker_dict)

    def test_read_pandas_format(self):
        test_data = {'file_id': ['file1', 'file2', 'file3'],
                     'text': ['This is a test', 'This is another test', 'A third test'],
                     'speaker_id': ['speaker1', 'speaker2', 'speaker3']}
        df = pd.DataFrame(test_data)

        with tempfile.NamedTemporaryFile(mode='w+') as f:
            csv_path = Path(f.name)
            df.to_csv(csv_path, sep='\t', encoding='utf-8', index=False)

            # Test the function with the created file
            text_dict, speaker_dict = read_pandas_format(csv_path)
            self.assertEqual({'file1': 'This is a test', 'file2': 'This is another test', 'file3': 'A third test'}, text_dict)
            self.assertEqual({'file1': 'speaker1', 'file2': 'speaker2', 'file3': 'speaker3'}, speaker_dict)