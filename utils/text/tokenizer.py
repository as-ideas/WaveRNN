from typing import List

from utils.text.symbols import phonemes, silent_phonemes
import numpy as np

class Tokenizer:

    def __init__(self) -> None:
        self.symbol_to_id = {s: i for i, s in enumerate(phonemes)}
        self.id_to_symbol = {i: s for i, s in enumerate(phonemes)}

    def __call__(self, text: str) -> np.array:

        text = [t for t in text if t in phonemes]
        non_silent = [t for t in text if t not in silent_phonemes]
        output = np.zeros((5, len(non_silent)+1))

        index = -1
        sil_index = 1
        for i, t in enumerate(text):
            if t in silent_phonemes and index == -1:
                print(t, text[:10])
            if t in silent_phonemes and index >= 0:
                if sil_index < 5:
                    output[sil_index][index] = self.symbol_to_id[t]
                    sil_index += 1
            else:
                index += 1
                output[0][index] = self.symbol_to_id[t]
                sil_index = 1

        return output


    def decode(self, seq: np.array) -> str:
        text = ''
        for t in range(seq.shape[-1]):
            for t_s in range(5):
                if seq[t_s, t] != 0:
                    text = text + self.id_to_symbol[seq[t_s, t]]
        return text



if __name__ == '__main__':
    text = '  (abc- (abc.'

    tokenizer = Tokenizer()
    toks = tokenizer(text)

    print(toks)

    dec = tokenizer.decode(toks)

    print(dec)