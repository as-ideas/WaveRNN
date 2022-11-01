import re
from typing import Dict, Any

from phonemizer.backend import EspeakBackend
from unidecode import unidecode

from utils.text.numbers import normalize_numbers
from utils.text.symbols import phonemes_set

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def no_cleaners(text):
    return text


def english_cleaners(text):
    text = unidecode(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    return text


class Cleaner:

    def __init__(self,
                 cleaner_name: str,
                 use_phonemes: bool,
                 lang: str) -> None:
        if cleaner_name == 'english_cleaners':
            self.clean_func = english_cleaners
        elif cleaner_name == 'no_cleaners':
            self.clean_func = no_cleaners
        else:
            raise ValueError(f'Cleaner not supported: {cleaner_name}! '
                             f'Currently supported: [\'english_cleaners\', \'no_cleaners\']')
        self.use_phonemes = use_phonemes
        self.lang = lang
        if use_phonemes:
            self.backend = EspeakBackend(language=lang,
                                         preserve_punctuation=True,
                                         with_stress=False,
                                         punctuation_marks=';:,.!?¡¿—…"«»“”()',
                                         language_switch='remove-flags')

    def __call__(self, text: str) -> str:
        text = self.clean_func(text)
        if self.use_phonemes:
            text = self.backend.phonemize([text], strip=True)[0]
            text = ''.join([p for p in text if p in phonemes_set])
        text = collapse_whitespace(text)
        text = text.strip()
        return text

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Cleaner':
        return Cleaner(
            cleaner_name=config['preprocessing']['cleaner_name'],
            use_phonemes=config['preprocessing']['use_phonemes'],
            lang=config['preprocessing']['language']
        )

