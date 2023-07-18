from dp.phonemizer import Phonemizer





if __name__ == '__main__':

    text = 'Verteidigungsminister Boris Pistorius (63, SPD) treibt die Beschaffung voran.'
    phonemizer = Phonemizer.from_checkpoint('/Users/cschaefe/workspace/tts-synthv3/app/11111111/models/bild_voice/phon_model/model.pt')

