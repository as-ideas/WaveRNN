import pandas as pd
import spacy
import tqdm
from dp.phonemizer import Phonemizer
from dp.utils.io import pickle_binary

from streamtts.pipeline.pipeline_factory import GERMAN_SPACY_MODEL
from streamtts.text.cleaners import TextCleaner, GermanExpander
from streamtts.text.forward_tacotron.symbols import phonemes_set
from streamtts.text.tokenizer import ForwardTacoTokenizer

if __name__ == '__main__':

    phonemizer = Phonemizer.from_checkpoint('/Users/cschaefe/workspace/tts-synthv3/app/11111111/models/bild_voice/phon_model/model.pt')
    alphabet = iter('abcdefghijklmnopqrstuvwxyz')

    tag_map = {'WHITESPACE': '0'}

    nlp = spacy.load(GERMAN_SPACY_MODEL)
    cleaner = TextCleaner(expanders={'de': GermanExpander(nlp)})
    df = pd.read_csv('/Users/cschaefe/datasets/nlp/welt_articles_phonemes.tsv', sep='\t', encoding='utf-8')

    rows = []

    tokenizer = ForwardTacoTokenizer()
    for text in tqdm.tqdm(df['text'], total=len(df)):
        try:
            cleaned = cleaner(text)
            doc = nlp(cleaned)
            tokens = [t for t in doc]
            pos = [t.pos_ for t in doc]
            text_phon = ''
            text_pos = ''
            token_texts = [t.text for t in tokens]
            phon_list = phonemizer.phonemise_list(token_texts, lang='de').phonemes

            for t, phons in zip(tokens, phon_list):
                if t.pos_ in tag_map:
                    pos_alpha = tag_map[t.pos_]
                else:
                    pos_alpha = next(alphabet)
                    tag_map[t.pos_] = pos_alpha
                text_phon += phons
                text_pos += pos_alpha * len(phons)
                if len(t.whitespace_) > 0:
                    text_phon += ' '
                    text_pos += tag_map['WHITESPACE']

            text_phon = tokenizer.decode(tokenizer(text_phon))
            if len(text_phon) == len(text_pos):
                row = {'text': text, 'text_phonemized': text_phon, 'text_phonemized_pos': text_pos}
                rows.append(row)

            if len(rows) % 100 == 0:
                df_new = pd.DataFrame(rows)
                df_new.to_csv('/Users/cschaefe/datasets/nlp/pos/alpha_pos.tsv', sep='\t', encoding='utf-8')
                pickle_binary(tag_map, '/Users/cschaefe/datasets/nlp/pos/alpha_pos_tag_map.pkl')
        except Exception as e:
            print(e)