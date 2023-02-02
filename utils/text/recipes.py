from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import pandas as pd
import tqdm

from utils.files import get_files

DEFAULT_SPEAKER_NAME = 'default_speaker'


def read_metadata(path: Path,
                  metafile: str,
                  format: str,
                  n_workers=1) -> Tuple[dict, dict]:
    if format == 'ljspeech':
        return read_ljspeech_format(path/metafile, multispeaker=False)
    elif format == 'ljspeech_multi':
        return read_ljspeech_format(path/metafile, multispeaker=True)
    elif format == 'vctk':
        return read_vctk_format(path, n_workers=n_workers)
    elif format == 'pandas':
        return read_pandas_format(path/metafile)
    else:
        raise ValueError(f'Metafile has unexpected ending: {path.stem}, expected [.csv, .tsv]"')


def read_ljspeech_format(path: Path, multispeaker: bool = False) -> Tuple[dict, dict]:
    if not path.is_file():
        raise ValueError(f'Could not find metafile: {path}, '
                         f'please make sure that you set the correct path and metafile name!')
    text_dict = {}
    speaker_dict = {}
    with open(str(path), encoding='utf-8') as f:
        for line in f:
            split = line.split('|')
            speaker_name = split[-2] if multispeaker and len(split) > 2 else DEFAULT_SPEAKER_NAME
            file_id, text = split[0], split[-1]
            text_dict[file_id] = text.replace('\n', '')
            speaker_dict[file_id] = speaker_name
    return text_dict, speaker_dict


def read_vctk_format(path: Path,
                     n_workers: int,
                     extension='.txt') -> Tuple[dict, dict]:
    files = get_files(path, extension=extension)
    text_dict = {}
    speaker_dict = {}
    pool = Pool(processes=n_workers)
    for i, (file, text) in tqdm.tqdm(enumerate(pool.imap_unordered(read_line, files), 1), total=len(files)):
        text_id = file.name.replace(extension, '')
        speaker_id = file.parent.stem
        text_dict[text_id] = text.replace('\n', '')
        speaker_dict[text_id] = speaker_id
    return text_dict, speaker_dict


def read_pandas_format(path: Path) -> Tuple[dict, dict]:
    if not path.is_file():
        raise ValueError(f'Could not find metafile: {path}, '
                         f'please make sure that you set the correct path and metafile name!')
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
