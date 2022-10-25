import math
from multiprocessing.pool import Pool

from utils.display import progbar, stream
from utils.files import get_files
from pathlib import Path
from typing import Union, Tuple
import pandas as pd

def get_value(row: pd.Series, key: str, default_value: float = 1) -> float:
    val = row.get(key, default_value)
    if math.isnan(val) or math.isinf(val):
        val = default_value
    return val

def ljspeech(path: Union[str, Path]):
    csv_files = get_files(path, extension='.csv')
    text_dict = {}
    text_prob = {}
    speaker_dict = {}
    for csv_file in csv_files:
        with open(str(csv_file), encoding='utf-8') as f:
            for line in f:
                split = line.split('|')
                text_dict[split[0]] = split[-1]
                text_prob[split[0]] = float(split[1])
                speaker_dict[split[0]] = split[2]
    return text_dict, speaker_dict, text_prob


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


def vctk(path: Union[str, Path], n_workers, extension='.txt') -> Tuple[dict, dict]:
    files = list(Path(path).glob('**/*' + extension))
    text_dict = {}
    speaker_id_dict = {}
    pool = Pool(processes=n_workers)
    for i, (file, text) in enumerate(pool.imap_unordered(read_line, files), 1):
        bar = progbar(i, len(files))
        message = f'{bar} {i}/{len(files)} '
        text_id = file.name.replace(extension, '')
        speaker_id = file.parent.stem
        text_dict[text_id] = text
        speaker_id_dict[text_id] = speaker_id
        stream(message)
    return text_dict, speaker_id_dict


def read_line(file: Path) -> Tuple[Path, str]:
    with open(str(file), encoding='utf-8') as f:
        line = f.readlines()[0]
    return file, line

