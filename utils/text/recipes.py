from pathlib import Path
from typing import Tuple
import pandas as pd

DEFAULT_SPEAKER_NAME = 'default_speaker'


def read_metadata(path: Path, multispeaker: bool = False) -> Tuple[dict, dict]:
    if not path.is_file():
        raise ValueError(f'Could not find metafile: {path}, '
                         f'please make sure that you set the correct path and metafile name!')
    if str(path).endswith('csv'):
        return read_ljspeech_format(path, multispeaker)
    elif str(path).endswith('tsv'):
        return read_pandas_format(path, multispeaker)
    else:
        raise ValueError(f'Metafile has unexpected ending: {path.stem}, expected [.csv, .tsv]"')


def read_ljspeech_format(path: Path, multispeaker: bool = False) -> Tuple[dict, dict]:
    text_dict = {}
    speaker_dict = {}
    with open(str(path), encoding='utf-8') as f:
        for line in f:
            split = line.split('|')
            speaker_name = split[-2] if multispeaker and len(split) > 2 else DEFAULT_SPEAKER_NAME
            file_id, text = split[0], split[-1]
            text_dict[file_id] = text
            speaker_dict[file_id] = speaker_name
    return text_dict, speaker_dict


def read_pandas_format(path: Path, multispeaker: bool = False) -> Tuple[dict, dict]:
    df = pd.read_csv(str(path), sep='\t', encoding='utf-8')
    text_dict = {}
    speaker_dict = {}
    for index, row in df.iterrows():
        id = row['file_id']
        text_dict[id] = row['text']
        speaker_dict[id] = row['speaker_id'] if multispeaker else DEFAULT_SPEAKER_NAME
    return text_dict, speaker_dict
