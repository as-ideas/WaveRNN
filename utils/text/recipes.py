import math
from pathlib import Path
from typing import Tuple

import pandas as pd


def read_metadata(path: Path) -> Tuple[dict, dict]:
    if not path.is_file():
        raise ValueError(f'Could not find metafile: {path}, '
                         f'please make sure that you set the correct path and metafile name!')
    if str(path).endswith('csv'):
        return read_ljspeech(path)
    elif str(path).endswith('tsv'):
        return read_multispeaker(path)
    else:
        raise ValueError(f'Metafile has unexpected ending: {path.stem}, expected [.csv, .tsv]"')


def read_ljspeech(path: Path) -> Tuple[dict, dict]:
    text_dict = {}
    speaker_dict = {}
    with open(str(path), encoding='utf-8') as f:
        for line in f:
            split = line.split('|')
            file_id, text = split[0], split[1]
            text_dict[file_id] = text
            speaker_dict[file_id] = 'singlespeaker'
    return text_dict, speaker_dict


def read_multispeaker(path: Path) -> Tuple[dict, dict]:
    df = pd.read_csv(str(path), sep='\t', encoding='utf-8')
    text_dict = {}
    speaker_dict = {}
    for index, row in df.iterrows():
        id = row['file_id']
        text_dict[id] = row['text']
        speaker_dict[id] = row['speaker_id']
    return text_dict, speaker_dict


def read_line(file: Path) -> Tuple[Path, str]:
    with open(str(file), encoding='utf-8') as f:
        line = f.readlines()[0]
    return file, line


def get_value(row: pd.Series, key: str, default_value: float = 1) -> float:
    val = row.get(key, default_value)
    if math.isnan(val) or math.isinf(val):
        val = default_value
    return val
