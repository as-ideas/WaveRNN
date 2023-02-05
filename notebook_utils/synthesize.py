import torch
import numpy as np
from typing import Callable

from utils.checkpoints import init_tts_model
from utils.dsp import DSP
from utils.text.cleaners import Cleaner
from utils.text.tokenizer import Tokenizer


class Synthesizer:

    def __init__(self,
                 tts_path: str,
                 device='cuda'):
        self.device = torch.device(device)
        tts_checkpoint = torch.load(tts_path, map_location=self.device)
        tts_config = tts_checkpoint['config']
        tts_model = init_tts_model(tts_config)
        tts_model.load_state_dict(tts_checkpoint['model'])
        self.tts_model = tts_model
        self.melgan = torch.hub.load('seungwonpark/melgan', 'melgan')
        self.melgan.to(device).eval()
        self.cleaner = Cleaner.from_config(tts_config)
        self.tokenizer = Tokenizer()
        self.dsp = DSP.from_config(tts_config)

    def __call__(self,
                 text: str,
                 voc_model: str,
                 alpha=1.0,
                 pitch_function: Callable[[torch.tensor], torch.tensor] = lambda x: x,
                 energy_function: Callable[[torch.tensor], torch.tensor] = lambda x: x,
                 ) -> np.array:
        x = self.cleaner(text)
        x = self.tokenizer(x)
        x = torch.tensor(x).unsqueeze(0)
        gen = self.tts_model.generate(x,
                                      alpha=alpha,
                                      pitch_function=pitch_function,
                                      energy_function=energy_function)
        m = gen['mel_post'].cpu()
        if voc_model == 'griffinlim':
            wav = self.dsp.griffinlim(m.squeeze().numpy(), n_iter=32)
        else:
            m = m.cuda()
            with torch.no_grad():
                wav = self.melgan.inference(m).cpu().numpy()
        return wav
