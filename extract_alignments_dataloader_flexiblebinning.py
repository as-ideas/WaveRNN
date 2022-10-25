import argparse
import itertools
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.tacotron import Tacotron
from trainer.common import to_device
from utils.checkpoints import restore_checkpoint
from utils.dataset import get_tts_datasets, BinnedLengthSampler, get_taco_duration_extraction_generator, filter_max_len
from utils.duration_extractor import DurationExtractor
from utils.files import read_config, unpickle_binary
from utils.metrics import attention_score
from utils.paths import Paths
from utils.text.tokenizer import Tokenizer


@dataclass
class DurationResult:
    item_id: str
    att_score: float
    align_score: float
    durs: Optional[np.array] = None


class DurationCollator:

    def __call__(self, x: List[DurationResult]) -> DurationResult:
        if len(x) > 1:
            raise ValueError(f'Batch size should be 1! Found dataset output wiht len: {len(x)}')
        return x[0]


class DurationDataset(Dataset):

    def __init__(self,
                 duration_extractor: DurationExtractor,
                 paths: Paths,
                 dataset_ids: List[str],
                 text_dict: Dict[str, str],
                 tokenizer: Tokenizer):
        self.metadata = dataset_ids
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.text_dict = text_dict
        self.duration_extractor = duration_extractor
        self.paths = paths

    def __getitem__(self, index: int) -> DurationResult:
        item_id = self.metadata[index]
        try:
            x = self.text_dict[item_id]
            x = self.tokenizer(x)
            mel = np.load(self.paths.mel / f'{item_id}.npy')
            mel = torch.from_numpy(mel)
            x = torch.tensor(x)
            att_npy = np.load(str(self.paths.att_pred / f'{item_id}.npy'))
            att = torch.from_numpy(att_npy)
            mel_len = mel.shape[-1]
            mel_len = torch.tensor(mel_len).unsqueeze(0)
            align_score, _ = attention_score(att.unsqueeze(0), mel_len, r=1)
            align_score = float(align_score)
            durs, att_score = self.duration_extractor(x=x, mel=mel, att=att)
            att_score = float(att_score)
            durs_npy = durs.cpu().numpy()
            assert np.sum(durs_npy) == mel_len, f'WARNINNG: Sum of durations did not match mel length for item {item_id}!'
        except Exception as e:
            print(e)
            return DurationResult(item_id=item_id, att_score=0.,
                                  align_score=0., durs=None)
        return DurationResult(item_id=item_id, att_score=att_score,
                              align_score=align_score, durs=durs_npy)

    def __len__(self):
        return len(self.metadata)


class DurationExtractorPipeline:

    def __init__(self,
                 paths: Paths,
                 config: Dict[str, Any],
                 duration_extractor: DurationExtractor) -> None:
        self.paths = paths
        self.config = config
        self.duration_extractor = duration_extractor

    def extract_attentions(self,
                           model: Tacotron,
                           max_batch_size: int = 1) -> None:
        assert model.r == 1, f'Model reduction factor is not one! Was: {model.r}'
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.eval()
        model.decoder.prenet.train()
        model.to(device)

        sum_att_score = 0

        train_data = unpickle_binary(paths.data/'train_dataset.pkl')
        val_data = unpickle_binary(paths.data/'val_dataset.pkl')
        dataset = train_data + val_data
        dataset_gen = get_taco_duration_extraction_generator(paths.data, dataset=dataset, max_batch_size=max_batch_size)

        pbar = tqdm(dataset_gen, total=len(dataset))
        sum_count = 0

        for i, batch in enumerate(pbar, 1):
            batch = to_device(batch, device=device)
            with torch.no_grad():
                _, _, att_batch = model(batch['x'], batch['mel'], batch['speaker_emb'])
            _, att_score = attention_score(att_batch, batch['mel_len'], r=1)
            sum_att_score += att_score.sum()
            B = batch['x_len'].size(0)
            sum_count += B
            for b in range(batch['x_len'].size(0)):
                x_len = batch['x_len'][b].cpu()
                mel_len = batch['mel_len'][b].cpu()
                item_id = batch['item_id'][b]
                att = att_batch[b, :mel_len, :x_len].cpu()
                np.save(paths.att_pred / f'{item_id}.npy', att.numpy(), allow_pickle=False)
            pbar.set_description(f'Avg attention score: {sum_att_score / sum_count}', refresh=True)

    def extract_durations(self,
                          num_workers: int = 0) -> None:
        train_set = unpickle_binary(paths.data / 'train_dataset.pkl')
        val_set = unpickle_binary(paths.data / 'val_dataset.pkl')
        text_dict = unpickle_binary(paths.data / 'text_dict.pkl')
        dataset = train_set + val_set
        dataset = [(file_id, mel_len) for file_id, mel_len in dataset
                   if (self.paths.att_pred / f'{file_id}.npy').is_file()]
        len_orig = len(dataset)
        data_ids, mel_lens = list(zip(*dataset))
        print(f'Found {len(data_ids)} / {len_orig} alignment files in {self.paths.att_pred}')
        att_score_dict = {}
        sum_att_score = 0

        dataset = DurationDataset(
            duration_extractor=self.duration_extractor,
            paths=paths, dataset_ids=data_ids,
            text_dict=text_dict, tokenizer=Tokenizer())

        dataset = DataLoader(dataset=dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=False,
                             collate_fn=DurationCollator(),
                             sampler=BinnedLengthSampler(lengths=mel_lens, batch_size=1, bin_size=num_workers * 8),
                             num_workers=num_workers)

        pbar = tqdm(dataset, total=len(dataset), smoothing=0.01)

        for i, res in enumerate(pbar, 1):
            pbar.set_description(f'Avg tuned attention score: {sum_att_score / i}', refresh=True)
            att_score_dict[res.item_id] = (res.align_score, res.att_score)
            sum_att_score += res.att_score
            if res.durs is not None:
                np.save(paths.alg / f'{res.item_id}.npy', res.durs, allow_pickle=False)


def batchify(input: List[Any], batch_size: int) -> List[List[Any]]:
    l = len(input)
    output = []
    for i in range(0, l, batch_size):
        batch = input[i:min(i + batch_size, l)]
        output.append(batch)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ForwardTacotron TTS')
    parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams.')
    args = parser.parse_args()
    config = read_config(args.config)
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])

    print('\nInitialising Tacotron Model...\n')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Tacotron.from_config(config).to(device)

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint(model=model, optim=optimizer,
                       path=paths.taco_checkpoints / 'latest_model.pt',
                       device=device)

    model.eval()
    model.decoder.prenet.train()

    duration_extractor = DurationExtractor(
        silence_threshold=config['preprocessing']['silence_threshold'],
        silence_prob_shift=config['preprocessing']['silence_prob_shift'])

    dur_pipeline = DurationExtractorPipeline(paths=paths,
                                             config=config,
                                             duration_extractor=duration_extractor)

    print('Extracting attention from tacotron...')
    dur_pipeline.extract_attentions(max_batch_size=32, model=model)
    print('Extracting durations from attention matrices...')
    dur_pipeline.extract_durations(num_workers=12)