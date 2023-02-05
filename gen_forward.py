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

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', default=None, type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--checkpoint', type=str, default=None, help='[string/path] path to .pt model file.')
    parser.add_argument('--config', metavar='FILE', default='default.yaml', help='The config containing all hyperparams. Only'
                                                                                'used if no checkpoint is set.')
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
        checkpoint_path = paths.forward_checkpoints / 'latest_model.pt'

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    tts_model = init_tts_model(config)
    tts_model.load_state_dict(checkpoint['model'])
    speaker_embedding = None
    if args.speaker is not None:
        assert 'speaker_embeddings' in checkpoint, 'Could not find speaker embeddings in checkpoint! Make sure you ' \
                                                   'use trained multispeaker model!'
        speaker_embeddings = checkpoint.get('speaker_embeddings', None)
        assert args.speaker in speaker_embeddings, \
            f'Provided speaker not found in speaker embeddings: {args.speaker},\n' \
            f'Available speakers: {checkpoint["speaker_embeddings"].keys()}'
        speaker_embedding = speaker_embeddings[args.speaker]

    print(f'Initialized tts model: {tts_model}')
    print(f'Restored model with step {tts_model.get_step()}')
    dsp = DSP.from_config(config)

    voc_model, voc_dsp = None, None
    out_path = Path('model_outputs')
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tts_model.to(device)
    cleaner = Cleaner.from_config(config)
    tokenizer = Tokenizer()

    print(f'Using device: {device}\n')
    if args.input_text:
        texts = [args.input_text]
    else:
        with open('sentences.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()

    tts_k = tts_model.get_step() // 1000
    tts_model.eval()

    simple_table([('Forward Tacotron', str(tts_k) + 'k'),
                  ('Vocoder Type', args.vocoder)])

    # simple amplification of pitch
    pitch_function = lambda x: x * args.amp
    energy_function = lambda x: x

    for i, x in enumerate(texts, 1):
        print(f'\n| Generating {i}/{len(texts)}')
        text = x
        x = cleaner(x)
        x = tokenizer(x)
        x = torch.as_tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        speaker_name = args.speaker if args.speaker is not None else 'default_speaker'
        wav_name = f'{i}_forward_{tts_k}k_{speaker_name}_alpha{args.alpha}_amp{args.amp}_{args.vocoder}'

        input = {
            'x': x,
            'alpha': args.alpha,
            'pitch_function': pitch_function,
            'energy_function': energy_function
        }
        if speaker_embedding is not None:
            input.update({'speaker_emb': speaker_embedding})

        gen = tts_model.generate(**input)

        m = gen['mel_post'].cpu()
        if args.vocoder == 'melgan':
            torch.save(m, out_path / f'{wav_name}.mel')
        if args.vocoder == 'hifigan':
            np.save(str(out_path / f'{wav_name}.npy'), m.numpy(), allow_pickle=False)
        elif args.vocoder == 'griffinlim':
            wav = dsp.griffinlim(m.squeeze().numpy())
            dsp.save_wav(wav, out_path / f'{wav_name}.wav')

    print('\n\nDone.\n')