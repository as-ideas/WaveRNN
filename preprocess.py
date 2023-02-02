import warnings
# Ignore future warnings by librosa in resemblyzer
from collections import Counter
from typing import Tuple

from utils.text.recipes import read_ljspeech_format, read_metadata

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import traceback
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from random import Random

import tqdm
import torch
from resemblyzer import VoiceEncoder
from resemblyzer import preprocess_wav as preprocess_resemblyzer

from pitch_extraction.pitch_extractor import PitchExtractor, new_pitch_extractor_from_config
from utils.display import *
from utils.dsp import *
from utils.files import get_files, pickle_binary, read_config
from utils.paths import Paths
from utils.text.cleaners import Cleaner


SPEAKER_EMB_DIM = 256  # fixed speaker dim from VoiceEncoder


def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n


@dataclass
class DataPoint:
    item_id: str
    mel_len: int
    text: str
    mel: np.array
    pitch: np.array
    reference_wav: np.array


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

    def __call__(self, id_path: Tuple[str, Path]) -> Union[DataPoint, None]:
        item_id, path = id_path
        try:
            dp = self._convert_file(item_id, path)
            np.save(self.paths.mel/f'{dp.item_id}.npy', dp.mel, allow_pickle=False)
            np.save(self.paths.raw_pitch/f'{dp.item_id}.npy', dp.pitch, allow_pickle=False)
            return dp
        except Exception as e:
            print(traceback.format_exc())
            return None

    def _convert_file(self, item_id: str, path: Path) -> DataPoint:
        y = self.dsp.load_wav(path)
        reference_wav = preprocess_resemblyzer(y, source_sr=self.dsp.sample_rate)
        if self.dsp.should_trim_long_silences:
           y = self.dsp.trim_long_silences(y)
        if self.dsp.should_trim_start_end_silence:
           y = self.dsp.trim_silence(y)
        peak = np.abs(y).max()
        if self.dsp.should_peak_norm or peak > 1.0:
            y /= peak
            y = y * 0.95
        mel = self.dsp.wav_to_mel(y)
        pitch = self.pitch_extractor(y)
        text = self.text_dict[item_id]
        text = self.cleaner(text)
        return DataPoint(item_id=item_id,
                         mel=mel.astype(np.float32),
                         mel_len=mel.shape[-1],
                         text=text,
                         pitch=pitch.astype(np.float32),
                         reference_wav=reference_wav)


parser = argparse.ArgumentParser(description='Dataset preprocessing')
parser.add_argument('--path', '-p', help='directly point to dataset')
parser.add_argument('--config', metavar='FILE', default='configs/singlespeaker.yaml',
                    help='The config containing all hyperparams.')
parser.add_argument('--metafile', '-m', default='metadata.csv',
                    help='name of the metafile in the dataset dir')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers,
                    default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
args = parser.parse_args()


if __name__ == '__main__':
    config = read_config(args.config)
    audio_format = config['preprocessing']['audio_format']
    audio_files = get_files(Path(args.path), audio_format)
    file_id_to_audio = {w.name.replace(audio_format, ''): w for w in audio_files}
    audio_ids = set(file_id_to_audio.keys())
    paths = Paths(config['data_path'], config['tts_model_id'])
    n_workers = max(1, args.num_workers)

    print(f'\nFound {len(audio_files)} {audio_format} files in "{args.path}".')
    assert len(audio_files) > 0, f'Found no {audio_format} files in {args.path}, exiting.'

    print('Preparing metadata...')
    text_dict, speaker_dict_raw = read_metadata(path=Path(args.path),
                                                metafile=args.metafile,
                                                format=config['preprocessing']['metafile_format'],
                                                n_workers=n_workers)
    text_dict = {item_id: text for item_id, text in text_dict.items()
                 if item_id in audio_ids and len(text) > config['preprocessing']['min_text_len']}
    file_id_to_audio = {k: v for k, v in file_id_to_audio.items() if k in text_dict}
    speaker_dict = {item_id: speaker for item_id, speaker in speaker_dict_raw.items() if item_id in audio_ids}
    speaker_counts = Counter(speaker_dict.values())

    assert len(file_id_to_audio) > 0, f'No audio file is indexed in metadata, exiting. ' \
                                      f'Pease make sure the audio ids match the ids in the metadata. ' \
                                      f'\nAudio ids: {sorted(list(audio_ids))[:5]}... ' \
                                      f'\nText ids: {sorted(list(speaker_dict_raw.keys()))[:5]}...'
    print(f'Will use {len(file_id_to_audio)} {audio_format} files that are indexed in metafile.\n')
    print(f'\n{"Speaker":30} {"Count":5}')
    print(f'------------------------------|-------')
    for speaker, count in speaker_counts.most_common():
        print(f'{speaker:30} {count:5}')
    print()

    dsp = DSP.from_config(config)
    nval = config['preprocessing']['n_val']

    if nval > len(file_id_to_audio):
        nval = len(file_id_to_audio) // 5
        print(f'\nWARNING: Using nval={nval} since the preset nval exceeds number of training files.')

    simple_table([
        ('Sample Rate', dsp.sample_rate),
        ('Hop Length', dsp.hop_length),
        ('CPU Usage', f'{n_workers}/{cpu_count()}'),
        ('Num Validation', nval),
        ('Pitch Extraction', config['preprocessing']['pitch_extractor'])
    ])

    pool = Pool(processes=n_workers)
    dataset = []
    cleaned_texts = []
    cleaner = Cleaner.from_config(config)
    pitch_extractor = new_pitch_extractor_from_config(config)

    preprocessor = Preprocessor(paths=paths,
                                text_dict=text_dict,
                                dsp=dsp,
                                pitch_extractor=pitch_extractor,
                                cleaner=cleaner,
                                lang=config['preprocessing']['language'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    voice_encoder = VoiceEncoder().to(device)
    file_id_audio_list = list(file_id_to_audio.items())
    successful_ids = set()

    for i, dp in tqdm.tqdm(enumerate(pool.imap_unordered(preprocessor, file_id_audio_list), 1),
                           total=len(audio_files), smoothing=0.1):
        if dp is not None and dp.item_id in text_dict:
            try:
                emb = voice_encoder.embed_utterance(dp.reference_wav)
                np.save(paths.speaker_emb / f'{dp.item_id}.npy', emb, allow_pickle=False)
                dataset += [(dp.item_id, dp.mel_len)]
                cleaned_texts += [(dp.item_id, dp.text)]
                successful_ids.add(dp.item_id)
            except Exception as e:
                print(traceback.format_exc())

    # filter according to successfully preprocessed datapoints
    text_dict = {k: v for k, v in text_dict.items() if k in successful_ids}
    speaker_dict = {k: v for k, v in speaker_dict.items() if k in successful_ids}
    speaker_counts = Counter(speaker_dict.values())

    # create stratified train / val split
    dataset.sort()
    random = Random(42)
    random.shuffle(dataset)
    val_ratio = nval / len(dataset)
    desired_val_counts = {speaker: max(count * val_ratio, 1) for speaker, count in speaker_counts.most_common()}
    val_speaker_counts = Counter()
    train_dataset, val_dataset = [], []
    for file_id, mel_len in dataset:
        speaker = speaker_dict[file_id]
        if val_speaker_counts.get(speaker, 0) < desired_val_counts[speaker]:
            val_dataset.append((file_id, mel_len))
            val_speaker_counts.update([speaker])
        else:
            train_dataset.append((file_id, mel_len))

    # sort val dataset longest to shortest
    val_dataset.sort(key=lambda d: -d[1])
    text_dict = {id: text for id, text in cleaned_texts}
    pickle_binary(text_dict, paths.text_dict)
    pickle_binary(speaker_dict, paths.speaker_dict)
    pickle_binary(train_dataset, paths.train_dataset)
    pickle_binary(val_dataset, paths.val_dataset)

    print('Averaging speaker embeddings...')

    mean_speaker_embs = {speaker: np.zeros(SPEAKER_EMB_DIM, dtype=float) for speaker in speaker_dict.values()}
    for file_id, speaker in tqdm.tqdm(speaker_dict.items(), total=len(speaker_dict), smoothing=0.1):
        emb = np.load(paths.speaker_emb / f'{file_id}.npy')
        mean_speaker_embs[speaker] += emb
    for speaker, emb in mean_speaker_embs.items():
        emb = emb / speaker_counts[speaker]
        emb = emb / np.linalg.norm(emb, 2)
        np.save(paths.mean_speaker_emb / f'{speaker}.npy', emb, allow_pickle=False)

    print('\n\nCompleted. Ready to run "python train_tacotron.py". \n')
