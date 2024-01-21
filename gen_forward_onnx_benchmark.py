import argparse
from pathlib import Path
import numpy as np
import onnxruntime
from utils.files import read_config
from utils.dsp import DSP
from utils.paths import Paths
from utils.text.cleaners import Cleaner
from utils.text.tokenizer import Tokenizer
import time

if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--checkpoint', type=str, default=None, help='[string/path] path to .onnx model file.')
    parser.add_argument('--config', metavar='FILE', default='default.yaml', help='The config containing all hyperparams.')
    parser.add_argument('--speaker', type=str, default=None, help='Speaker to generate audio for (only multispeaker).')

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        config = read_config(args.config)
        paths = Paths(config['data_path'], config['tts_model_id'])
        checkpoint_path = paths.forward_checkpoints / 'latest_model.onnx'
    sess_options = onnxruntime.SessionOptions()
    checkpoint = onnxruntime.InferenceSession(str(checkpoint_path), sess_options=sess_options)
    config = read_config(args.config)
    dsp = DSP.from_config(config)
    cleaner = Cleaner.from_config(config)
    tokenizer = Tokenizer()

    with open('sentences.txt', 'r', encoding='utf-8') as f:
        texts = f.readlines()
    for i, x in enumerate(texts, 1):
        print(f'\n| Generating {i}/{len(texts)}')
        text = x
        x = cleaner(x)
        x = tokenizer(x)
        text = np.expand_dims(np.array(x, dtype=np.int64), 0)
        synth_options = np.array(
            [1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        speaker_name = args.speaker if args.speaker is not None else 'default_speaker'
        start_time = time.perf_counter()
        m = checkpoint.run(
            None,
            {
                "input": text,
                "synth_options": synth_options,
            },
        )[0]
        end_time = time.perf_counter()
        mel_length = m.shape[-1]
        spec_length = mel_length * dsp.hop_length
        spec_sec = spec_length / dsp.sample_rate
        infer_sec = (end_time - start_time)
        rtf = infer_sec / spec_sec*1000
        print(f"Sentence {i} generation time: {infer_sec} MS, RTF: {rtf} MS.")

    print('\n\nDone.\n')