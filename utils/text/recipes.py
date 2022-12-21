import math
from utils.files import get_files
from pathlib import Path
from typing import Union, Tuple
import pandas as pd


def ljspeech(path: Union[str, Path]):
    csv_file = get_files(path, extension='.csv')
    assert len(csv_file) == 1
    text_dict = {}
    with open(str(csv_file[0]), encoding='utf-8') as f:
        for line in f:
            split = line.split('|')
            text_dict[split[0]] = split[-1]
    return text_dict


def multispeaker(path: Union[str, Path]):
    df = pd.read_csv(path, sep='\t', encoding='utf-8')

    text_dict = {}
    text_prob = {}
    text_sim = {}
    speaker_dict = {}

    for index, row in df.iterrows():
        id = row['file_id']
        text_dict[id] = row['text_phonemized']
        text_prob[id] = get_value(row, 'transcription_probability', default_value=1)
        text_sim[id] = get_value(row, 'levenshtein_similarity', default_value=1)
        speaker_dict[id] = row['speaker_id'] + '_' + row['book_id']

    return text_dict, speaker_dict, text_prob, text_sim


def read_line(file: Path) -> Tuple[Path, str]:
    with open(str(file), encoding='utf-8') as f:
        line = f.readlines()[0]
    return file, line


def get_value(row: pd.Series, key: str, default_value: float = 1) -> float:
    val = row.get(key, default_value)
    if math.isnan(val) or math.isinf(val):
        val = default_value
    return val
