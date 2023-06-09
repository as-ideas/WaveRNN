import pandas as pd
import spacy
import re

def remove_text_in_carots(input_string):
    # Use regular expression to remove text within carots
    output_string = re.sub(r'<.*?>', '', input_string)
    return output_string

if __name__ == '__main__':
    df = pd.read_csv('/Users/cschaefe/datasets/tts-synth-data/welt/metadata.tsv', sep='\t', encoding='utf-8')

    nlp_de = spacy.load('de_dep_news_trf')

    pos_tags = set()

    for text, phons  in zip(df['text'], df['text_phonemized']):
        print()


        text_new = remove_text_in_carots(text)

        toks = nlp_de(text_new)
        toks = [t for t in toks if not t.is_punct]

        for t, tn, p, tok in zip(text.split(), text_new.split(), phons.split(), toks):

            print(t, tn, p, '|', tok.text, tok.pos_)
            pos_tags.add(tok.pos_)

            print(pos_tags, len(pos_tags))


