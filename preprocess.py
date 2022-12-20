import argparse
import traceback
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool, cpu_count
from random import Random

import tqdm

from pitch_extraction.pitch_extractor import PitchExtractor, LibrosaPitchExtractor, PyworldPitchExtractor
from utils.display import *
from utils.dsp import *
from utils.files import get_files, pickle_binary, read_config
from utils.paths import Paths
from utils.text.cleaners import Cleaner
from utils.text.recipes import ljspeech


class PitchExtractionMethod(Enum):
    LIBROSA = 'librosa'
    PYWORLD = 'pyworld'


def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n


@dataclass
class DataPoint:
    item_id: str = None
    mel_len: int = None
    text: str = None
    mel: np.array = None
    pitch: np.array = None


class Preprocessor:

    def __init__(self,
                 paths: Paths,
                 text_dict: Dict[str, str],
                 cleaner: Cleaner,
                 dsp: DSP,
                 pitch_extractor: PitchExtractor,
                 lang: str) -> None:
        self.paths = paths
        self.text_dict = text_dict
        self.cleaner = cleaner
        self.lang = lang
        self.dsp = dsp
        self.pitch_extractor = pitch_extractor

    def __call__(self, path: Path) -> Union[DataPoint, None]:
        try:
            dp = self._convert_file(path)
            np.save(self.paths.mel/f'{dp.item_id}.npy', dp.mel, allow_pickle=False)
            np.save(self.paths.raw_pitch/f'{dp.item_id}.npy', dp.pitch, allow_pickle=False)
            return dp
        except Exception as e:
            print(traceback.format_exc())
            return None

    def _convert_file(self, path: Path) -> DataPoint:
        y = self.dsp.load_wav(path)
        if self.dsp.should_trim_long_silences:
           y = self.dsp.trim_long_silences(y)
        if self.dsp.should_trim_start_end_silence:
           y = self.dsp.trim_silence(y)
        peak = np.abs(y).max()
        if self.dsp.should_peak_norm or peak > 1.0:
            y /= peak
        mel = self.dsp.wav_to_mel(y)
        pitch = self.pitch_extractor(y)
        print(pitch)
        item_id = path.stem
        text = self.text_dict[item_id]
        text = self.cleaner(text)

        return DataPoint(item_id=item_id,
                         mel=mel.astype(np.float32),
                         mel_len=mel.shape[-1],
                         text=text,
                         pitch=pitch.astype(np.float32))


parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', help='directly point to dataset path')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
parser.add_argument('--config', metavar='FILE', default='default.yaml', help='The config containing all hyperparams.')
args = parser.parse_args()


if __name__ == '__main__':

    config = read_config(args.config)
    wav_files = get_files(args.path, '.wav')
    wav_ids = {w.stem for w in wav_files}
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])
    print(f'\n{len(wav_files)} .wav files found in "{args.path}"')
    assert len(wav_files) > 0, f'Found no wav files in {args.path}, exiting.'

    text_dict = ljspeech(args.path)
    text_dict = {item_id: text for item_id, text in text_dict.items()
                 if item_id in wav_ids and len(text) > config['preprocessing']['min_text_len']}
    wav_files = [w for w in wav_files if w.stem in text_dict]
    print(f'Using {len(wav_files)} wav files that are indexed in metafile.\n')

    n_workers = max(1, args.num_workers)

    dsp = DSP.from_config(config)

    simple_table([
        ('Sample Rate', dsp.sample_rate),
        ('Hop Length', dsp.hop_length),
        ('CPU Usage', f'{n_workers}/{cpu_count()}'),
        ('Num Validation', config['preprocessing']['n_val'])
    ])

    pool = Pool(processes=n_workers)
    dataset = []
    cleaned_texts = []
    cleaner = Cleaner.from_config(config)
    preproc_config = config['preprocessing']
    pitch_extractor_type = preproc_config['pitch_extractor']
    if pitch_extractor_type == 'librosa':
        pitch_extractor = LibrosaPitchExtractor(fmin=preproc_config['pitch_min_freq'],
                                                fmax=preproc_config['pitch_max_freq'],
                                                frame_length=preproc_config['pitch_frame_length'],
                                                sample_rate=dsp.sample_rate,
                                                hop_length=dsp.hop_length)
    elif pitch_extractor_type == 'pyworld':
        pitch_extractor = PyworldPitchExtractor(hop_length=dsp.hop_length, sample_rate=dsp.sample_rate)
    else:
        raise ValueError(f'Invalid pitch extractor type: {pitch_extractor_type}, choices: [librosa, pyworld].')

    preprocessor = Preprocessor(paths=paths,
                                text_dict=text_dict,
                                dsp=dsp,
                                pitch_extractor=pitch_extractor,
                                cleaner=cleaner,
                                lang=preproc_config['language'])

    for i, dp in tqdm.tqdm(enumerate(pool.imap_unordered(preprocessor, wav_files), 1), total=len(wav_files)):
        if dp is not None and dp.item_id in text_dict:
            dataset += [(dp.item_id, dp.mel_len)]
            cleaned_texts += [(dp.item_id, dp.text)]

    dataset.sort()
    random = Random(42)
    random.shuffle(dataset)
    train_dataset = dataset[config['preprocessing']['n_val']:]
    val_dataset = dataset[:config['preprocessing']['n_val']]
    # sort val dataset longest to shortest
    val_dataset.sort(key=lambda d: -d[1])
    print(f'First val sample: {val_dataset[0][0]}')

    text_dict = {id: text for id, text in cleaned_texts}

    pickle_binary(text_dict, paths.data/'text_dict.pkl')
    pickle_binary(train_dataset, paths.data/'train_dataset.pkl')
    pickle_binary(val_dataset, paths.data/'val_dataset.pkl')

    print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
