import pandas as pd
import spacy
import re

import tqdm

from utils.text.symbols import phonemes, phonemes_set


def remove_text_in_carots(input_string):
    # Use regular expression to remove text within carots
    output_string = re.sub(r'<.*?>', '', input_string)
    return output_string

if __name__ == '__main__':
    df = pd.read_csv('/Users/cschaefe/datasets/tts-synth-data/bild/processed_metadata.tsv', sep='\t', encoding='utf-8')

    nlp_de = spacy.load('de_dep_news_trf')
    nlp_en = spacy.load('en_core_web_trf')

    pos_tags = set()

    rows = []

    false_pos = 0
    total = 0

    for id, text, phons  in tqdm.tqdm(zip(df['id'], df['text'], df['text_phonemized']), total=len(df)):
        print()
        total += 1
        phons = phons.split(' ')
        phons = ' '.join([p for p in phons if p not in {'-'}])
        phons = ''.join([p for p in phons if p in phonemes_set])
        phons = phons.strip()
        text_new = remove_text_in_carots(text)

        if id.startswith('bild_r_1'):
            toks = nlp_en(text_new)
        else:
            toks = nlp_de(text_new)

        toks = [t for t in toks if not t.is_punct]

        pos = [t.pos_ for t in toks]

        if len(phons.split()) - len(pos) != 0:
            false_pos += 1

        print(id, len(phons.split()), len(pos), 'fp', false_pos, 'tot', total)
        print(text)
        print(phons)
        print([t for t in toks])
        print('|'.join(pos))

        if len(phons.split()) - len(pos) == 0:
            rows.append({
                'file_id': id,
                'text:': text_new,
                'text_phonemized': phons,
                'pos': '|'.join(pos),
            })


        if total % 100 == 0:
            df_new = pd.DataFrame(rows)
            df_new.to_csv('/Users/cschaefe/datasets/nlp/bild_phons_pos.tsv', sep='\t', encoding='utf-8')