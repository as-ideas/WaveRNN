import random
import shutil
from collections import Counter
from itertools import groupby
from pathlib import Path
import librosa
import torch
import tqdm
from dp.utils.io import pickle_binary
from numpy.polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from utils.display import plot_pitch, plot_mel
from utils.dsp import DSP
from utils.files import unpickle_binary, read_config
from utils.paths import Paths
from utils.text.symbols import phonemes
from utils.text.tokenizer import Tokenizer

from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':

    res = 32

    for up, layers in zip([2, 2, 2, 2, 2, 2, 2, 2], [4, 5, 6, 6, 6, 7, 7, 8]):
        res *= up
        dil = [3**i for i in range(layers)]
        print(up, res, dil)