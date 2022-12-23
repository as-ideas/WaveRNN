from pathlib import Path
from typing import Tuple

DEFAULT_SPEAKER_NAME = 'singlespeaker'


def read_ljspeech_format(path: Path) -> Tuple[dict, dict]:
    text_dict = {}
    speaker_dict = {}
    with open(str(path), encoding='utf-8') as f:
        for line in f:
            split = line.split('|')
            speaker_name = split[1] if len(split) > 2 else DEFAULT_SPEAKER_NAME
            file_id, text = split[0], split[-1]
            text_dict[file_id] = text
            speaker_dict[file_id] = speaker_name
    return text_dict, speaker_dict


