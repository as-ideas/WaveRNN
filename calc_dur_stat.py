import pandas as pd
from pathlib import Path
import numpy as np
import tqdm

from models.multi_forward_tacotron import DurationNormalizer
from utils.files import unpickle_binary, pickle_binary
from utils.text.symbols import phonemes, phonemes_set
from utils.text.tokenizer import Tokenizer




if __name__ == '__main__':

    data_path = Path('/Users/cschaefe/datasets/multispeaker_welt_bild')
    dur_path = data_path / 'alg'
    text_path = data_path / 'text_dict.pkl'
    speaker_path = data_path / 'speaker_dict.pkl'
    text_dict = unpickle_binary(text_path)
    speaker_dict = unpickle_binary(speaker_path)

    duration_normalizer = DurationNormalizer(speaker_path, dur_path, text_path)
    pickle_binary(duration_normalizer, data_path / 'duration_normalizer.pkl')
    durn = unpickle_binary(data_path / 'duration_normalizer.pkl')
    exit()
    for id, text in text_dict.items():

        print(id)
        durs = np.load(str(dur_path / f'{id}.npy'))
        speaker_id =speaker_dict[id]
        dur_norm = duration_normalizer.normalize(speaker_id, durs, text)
        dur_denorm = duration_normalizer.denormalize(speaker_id, dur_norm, text)
        print('orig:', durs)
        print('norm:', dur_norm)
        print('denorm:', dur_denorm)

