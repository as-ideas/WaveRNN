import torch

from models.fatchord_version import WaveRNN
from models.forward_tacotron import ForwardTacotron
from utils import hparams as hp
from utils.text.symbols import phonemes
from utils.paths import Paths
import argparse
from utils.text import text_to_sequence, clean_text
from utils.display import simple_table
from utils.dsp import reconstruct_waveform, save_wav

if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--tts_weights', type=str, help='[string/path] Load in different FastSpeech weights')
    parser.add_argument('--save_attention', '-a', dest='save_attn', action='store_true', help='Save Attention Plots')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    parser.add_argument('--alpha', type=float, default=1., help='Parameter for controlling length regulator for speedup '
                                                                'or slow-down of generated speech, e.g. alpha=2.0 is double-time')
    parser.set_defaults(input_text=None)
    parser.set_defaults(weights_path=None)

    # name of subcommand goes to args.vocoder
    subparsers = parser.add_subparsers(dest='vocoder')

    wr_parser = subparsers.add_parser('wavernn', aliases=['wr'])
    wr_parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')
    wr_parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slow Unbatched Generation')
    wr_parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    wr_parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    wr_parser.add_argument('--voc_weights', type=str, help='[string/path] Load in different WaveRNN weights')
    wr_parser.set_defaults(batched=None)

    gl_parser = subparsers.add_parser('griffinlim', aliases=['gl'])
    gl_parser.add_argument('--iters', type=int, default=32, help='[int] number of griffinlim iterations')

    mg_parser = subparsers.add_parser('melgan', aliases=['mg'])

    args = parser.parse_args()

    if args.vocoder in ['griffinlim', 'gl']:
        args.vocoder = 'griffinlim'
    elif args.vocoder in ['wavernn', 'wr']:
        args.vocoder = 'wavernn'
    elif args.vocoder in ['melgan', 'mg']:
        args.vocoder = 'melgan'
    else:
        raise argparse.ArgumentError('Must provide a valid vocoder type!')

    hp.configure(args.hp_file)  # Load hparams from file
    # set defaults for any arguments that depend on hparams
    if args.vocoder == 'wavernn':
        if args.target is None:
            args.target = hp.voc_target
        if args.overlap is None:
            args.overlap = hp.voc_overlap
        if args.batched is None:
            args.batched = hp.voc_gen_batched

        batched = args.batched
        target = args.target
        overlap = args.overlap

    input_text = args.input_text
    tts_weights = args.tts_weights
    save_attn = args.save_attn

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    if args.vocoder == 'wavernn':
        print('\nInitialising WaveRNN Model...\n')
        # Instantiate WaveRNN Model
        voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                            fc_dims=hp.voc_fc_dims,
                            bits=hp.bits,
                            pad=hp.voc_pad,
                            upsample_factors=hp.voc_upsample_factors,
                            feat_dims=hp.num_mels,
                            compute_dims=hp.voc_compute_dims,
                            res_out_dims=hp.voc_res_out_dims,
                            res_blocks=hp.voc_res_blocks,
                            hop_length=hp.hop_length,
                            sample_rate=hp.sample_rate,
                            mode=hp.voc_mode).to(device)

        voc_load_path = args.voc_weights if args.voc_weights else paths.voc_latest_weights
        voc_model.load(voc_load_path)

    print('\nInitialising Forward TTS Model...\n')
    tts_model = ForwardTacotron(embed_dims=hp.forward_embed_dims,
                                num_chars=len(phonemes),
                                durpred_rnn_dims=hp.forward_durpred_rnn_dims,
                                durpred_conv_dims=hp.forward_durpred_conv_dims,
                                durpred_dropout=hp.forward_durpred_dropout,
                                rnn_dim=hp.forward_rnn_dims,
                                postnet_k=hp.forward_postnet_K,
                                postnet_dims=hp.forward_postnet_dims,
                                prenet_k=hp.forward_prenet_K,
                                prenet_dims=hp.forward_prenet_dims,
                                highways=hp.forward_num_highways,
                                dropout=hp.forward_dropout,
                                n_mels=hp.num_mels).to(device)

    tts_load_path = tts_weights if tts_weights else paths.forward_latest_weights
    tts_model.load(tts_load_path)

    if input_text:
        text = clean_text(input_text.strip())
        inputs = [text_to_sequence(text)]
    else:
        with open('sentences.txt') as f:
            inputs = [clean_text(l.strip()) for l in f]
        inputs = [text_to_sequence(t) for t in inputs]

    tts_k = tts_model.get_step() // 1000

    if args.vocoder == 'wavernn':
        voc_k = voc_model.get_step() // 1000
        simple_table([('Forward Tacotron', str(tts_k) + 'k'),
                    ('Vocoder Type', 'WaveRNN'),
                    ('WaveRNN', str(voc_k) + 'k'),
                    ('Generation Mode', 'Batched' if batched else 'Unbatched'),
                    ('Target Samples', target if batched else 'N/A'),
                    ('Overlap Samples', overlap if batched else 'N/A')])

    elif args.vocoder == 'griffinlim':
        simple_table([('Forward Tacotron', str(tts_k) + 'k'),
                      ('Vocoder Type', 'Griffin-Lim'),
                      ('GL Iters', args.iters)])

    elif args.vocoder == 'melgan':
        simple_table([('Forward Tacotron', str(tts_k) + 'k'),
                    ('Vocoder Type', 'MelGAN')])

    for i, x in enumerate(inputs, 1):

        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, _ = tts_model.generate(x, alpha=args.alpha)

        if args.vocoder == 'griffinlim':
            v_type = args.vocoder
        elif args.vocoder == 'wavernn' and args.batched:
            v_type = 'wavernn_batched'
        else:
            v_type = 'wavernn_unbatched'

        if input_text:
            save_path = paths.forward_output/f'{input_text[:10]}_{args.alpha}_{v_type}_{tts_k}k.wav'
        else:
            save_path = paths.forward_output/f'{i}_{v_type}_{tts_k}k_alpha{args.alpha}.wav'

        if args.vocoder == 'wavernn':
            m = torch.tensor(m).unsqueeze(0)
            voc_model.generate(m, save_path, batched, hp.voc_target, hp.voc_overlap, hp.mu_law)
        if args.vocoder == 'melgan':
            m = torch.tensor(m).unsqueeze(0)
            torch.save(m, paths.forward_output/f'{i}_{tts_k}_alpha{args.alpha}.mel')
        elif args.vocoder == 'griffinlim':
            wav = reconstruct_waveform(m, n_iter=args.iters)
            save_wav(wav, save_path)

    print('\n\nDone.\n')
