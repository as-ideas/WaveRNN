import numpy as np
import pandas as pd
import tqdm
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings

# init embedding
embedding = TransformerDocumentEmbeddings('bert-base-german-cased')

# create a sentence
sentence = Sentence('The grass is green .')

# embed the sentence
embedding.embed(sentence)

if __name__ == '__main__':
    df = pd.read_csv('/Users/cschaefe/datasets/tts-synth-data/bild/processed_metadata.tsv', sep='\t', encoding='utf-8')

    embedding_de = TransformerDocumentEmbeddings('bert-base-german-cased')
    embedding_en = TransformerDocumentEmbeddings('bert-base-uncased')

    # create a sentence
    sentence = Sentence('Hallo du sack!')

    # embed the sentence
    embedding.embed(sentence)

    for id, text in tqdm.tqdm(zip(df['id'], df['text']), total=len(df)):
        sent = Sentence(text)
        if id.startswith('r_1'):
            embedding_en.embed(sent)
        else:
            embedding_de.embed(sent)
        emb = sent.embedding.detach().numpy()
        print(emb.shape)
        np.save(f'/Users/cschaefe/workspace/ForwardTacotron/data/emb_bert/{id}.npy', emb, allow_pickle=False)