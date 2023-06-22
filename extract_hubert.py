from pathlib import Path
import torch
import librosa
import tqdm
from transformers import AutoProcessor, HubertModel
import soundfile as sf

device = torch.device('cpu')

processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


wavs = list(Path('/Users/cschaefe/datasets/Snippets').glob('**/*.wav'))

for wav_path in tqdm.tqdm(wavs, total=len(wavs)):
    id = wav_path.stem
    wav = librosa.load(wav_path, sr=16000)[0]
    input_values = processor(wav, return_tensors="pt", sampling_rate=16000).input_values.to(device)  # Batch size 1
    hidden_states = model(input_values).last_hidden_state
    torch.save(hidden_states, f'multispeaker_welt_bild/hubert/{id}.pt')
