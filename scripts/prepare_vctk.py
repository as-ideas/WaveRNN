import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Tuple

import tqdm

from utils.files import get_files


parser = argparse.ArgumentParser(description='Pepares VCTK metafile')
parser.add_argument('--path', '-p', help='directly point to dataset')
args = parser.parse_args()


def read_vctk_format(path: Union[str, Path],
                     n_workers,
                     extension='.txt') -> Tuple[dict, dict]:
    files = get_files(path, extension=extension)
    text_dict = {}
    speaker_dict = {}
    pool = Pool(processes=n_workers)
    for i, (file, text) in tqdm.tqdm(enumerate(pool.imap_unordered(read_line, files), 1), total=len(files)):
        text_id = file.name.replace(extension, '')
        speaker_id = file.parent.stem
        text_dict[text_id] = text
        speaker_dict[text_id] = speaker_id
    return text_dict, speaker_dict


def read_line(file: Path) -> Tuple[Path, str]:
    with open(str(file), encoding='utf-8') as f:
        line = f.readlines()[0]
    return file, line

if __name__ == '__main__':
    path = '/Users/cschaefe/datasets/VCTK'
    text_dict, speaker_dict = read_vctk_format(path, n_workers=5)
    print(text_dict)
    print(speaker_dict)