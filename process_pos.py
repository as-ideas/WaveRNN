import pandas as pd
import numpy as np
from utils.files import pickle_binary

if __name__ == '__main__':
    df = pd.read_csv('/Users/cschaefe/datasets/nlp/bild_phons_pos.tsv', sep='\t', encoding='utf-8')

    all_pos = set()

    for pos in df['pos']:
        all_pos.update(set(pos.split('|')))

    all_pos = sorted(list(all_pos))

    pos_num = {p: i for i, p in enumerate(all_pos, 2)}
    print(pos_num)

    text_dict = {}
    pos_dict = {}

    for id, phons, pos in zip(df['file_id'], df['text_phonemized'], df['pos']):
        phons = phons.split()
        pos = pos.split('|')

        pos_ids = []
        for p, pp in zip(phons, pos):
            ids = pos_num[pp]
            pos_ids.extend([ids]*len(p) + [1])

        phons = ' '.join(phons)
        pos_ids = pos_ids[:len(phons)]
        pos_ids = np.array(pos_ids)

        text_dict[id] = phons


        print(phons)
        print(pos_ids)

        np.save(f'/Users/cschaefe/datasets/nlp/pos/phon_pos/{id}.npy', pos_ids)
    pickle_binary(text_dict, '/Users/cschaefe/datasets/nlp/pos/text_dict_bild_pos.pkl')
    pickle_binary(pos_num, '/Users/cschaefe/datasets/nlp/pos/pos_dict.pkl')

