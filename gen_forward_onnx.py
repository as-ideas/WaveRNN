import argparse
from pathlib import Path
import numpy as np
import onnxruntime
from utils.display import simple_table
from utils.dsp import DSP
from utils.files import read_config
from utils.paths import Paths
from utils.text.cleaners import Cleaner
from utils.text.tokenizer import Tokenizer


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', default=None, type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--checkpoint', type=str, default=None, help='[string/path] path to .pt model file.')
    parser.add_argument('--config', metavar='FILE', default='default.yaml', help='The config containing all hyperparams.')
    parser.add_argument('--speaker', type=str, default=None, help='Speaker to generate audio for (only multispeaker).')

    parser.add_argument('--alpha', type=float, default=1., help='Parameter for controlling length regulator for speedup '
                                                                'or slow-down of generated speech, e.g. alpha=2.0 is double-time')
    parser.add_argument('--amp', type=float, default=1., help='Parameter for controlling pitch amplification')

    # name of subcommand goes to args.vocoder
    subparsers = parser.add_subparsers(dest='vocoder')
    gl_parser = subparsers.add_parser('griffinlim')
    mg_parser = subparsers.add_parser('melgan')
    hg_parser = subparsers.add_parser('hifigan')

    args = parser.parse_args()

    assert args.vocoder in {'griffinlim', 'melgan', 'hifigan'}, \
        'Please provide a valid vocoder! Choices: [griffinlim, melgan, hifigan]'

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        config = read_config(args.config)
        paths = Paths(config['data_path'], config['tts_model_id'])
        checkpoint_path = paths.forward_checkpoints / 'latest_model.onnx'
    sess_options = onnxruntime.SessionOptions()
    checkpoint = onnxruntime.InferenceSession(str(checkpoint_path), sess_options=sess_options)
    config = read_config(args.config)
    dsp = DSP.from_config(config)

    voc_model, voc_dsp = None, None
    out_path = Path('model_outputs')
    out_path.mkdir(parents=True, exist_ok=True)
    cleaner = Cleaner.from_config(config)
    tokenizer = Tokenizer()

    if args.input_text:
        texts = [args.input_text]
    else:
        with open('sentences.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()

    pitch_function = lambda x: x * args.amp
    energy_function = lambda x: x

    for i, x in enumerate(texts, 1):
        print(f'\n| Generating {i}/{len(texts)}')
        text = x
        x = cleaner(x)
        x = tokenizer(x)
        text = np.expand_dims(np.array(x, dtype=np.int64), 0)
        synth_options = np.array(
            [args.alpha, 1.0, 1.0],
            dtype=np.float32,
        )
        speaker_name = args.speaker if args.speaker is not None else 'default_speaker'
        wav_name = f"test{i}"
        m = checkpoint.run(
            None,
            {
                "input": text,
                "synth_options": synth_options,
            },
        )[0]
        #m = (m * 32767).astype(np.int16)
        if args.vocoder == 'melgan':
            torch.save(m, out_path / f'{wav_name}.mel')
        if args.vocoder == 'hifigan':
            np.save(str(out_path / f'{wav_name}.npy'), m, allow_pickle=False)
        elif args.vocoder == 'griffinlim':
            wav = dsp.griffinlim(m)
            dsp.save_wav(wav, out_path / f'{wav_name}.wav')

    print('\n\nDone.\n')