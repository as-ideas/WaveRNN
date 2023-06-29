import argparse
from pathlib import Path
import numpy as np
import torch
from utils.checkpoints import init_tts_model
from utils.display import simple_table
from utils.dsp import DSP
from utils.files import read_config
from utils.paths import Paths
from utils.text.cleaners import Cleaner
from utils.text.tokenizer import Tokenizer


if __name__ == '__main__':

    dur_files = list(Path('/Users/cschaefe/datasets/multispeaker_welt_bild/alg').glob('**/*.npy'))