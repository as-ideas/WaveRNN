import os
from pathlib import Path


class Paths:
    """Manages and configures the paths used by WaveRNN, Tacotron, and the data."""
    def __init__(self, data_path, tts_id):

        # directories
        self.base = Path(__file__).parent.parent.expanduser().resolve()
        self.data = Path(data_path).expanduser().resolve()
        self.quant = self.data/'quant'
        self.mel = self.data/'mel'
        self.gta = self.data/'gta'
        self.audio = self.data/'audio'
        self.att_pred = self.data/'att_pred'
        self.alg = self.data/'alg'
        self.speaker_emb = self.data/'speaker_emb'
        self.mean_speaker_emb = self.data/'mean_speaker_emb'
        self.raw_pitch = self.data/'raw_pitch'
        self.phon_pitch = self.data/'phon_pitch'
        self.phon_energy = self.data/'phon_energy'
        self.model_output = self.base / 'model_output'
        self.taco_checkpoints = self.base / 'checkpoints' / f'{tts_id}.tacotron'
        self.taco_log = self.taco_checkpoints / 'logs'
        self.forward_checkpoints = self.base/'checkpoints'/f'{tts_id}.forward'
        self.forward_log = self.forward_checkpoints/'logs'

        # pickle objects
        self.train_dataset = self.data / 'train_dataset.pkl'
        self.val_dataset = self.data / 'val_dataset.pkl'
        self.text_dict = self.data / 'text_dict.pkl'
        self.speaker_dict = self.data / 'speaker_dict.pkl'
        self.duration_stats = self.data / 'duration_stats.pkl'

        self.create_paths()

    def create_paths(self):
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.quant, exist_ok=True)
        os.makedirs(self.mel, exist_ok=True)
        os.makedirs(self.audio, exist_ok=True)
        os.makedirs(self.gta, exist_ok=True)
        os.makedirs(self.alg, exist_ok=True)
        os.makedirs(self.speaker_emb, exist_ok=True)
        os.makedirs(self.mean_speaker_emb, exist_ok=True)
        os.makedirs(self.att_pred, exist_ok=True)
        os.makedirs(self.raw_pitch, exist_ok=True)
        os.makedirs(self.phon_pitch, exist_ok=True)
        os.makedirs(self.phon_energy, exist_ok=True)
        os.makedirs(self.taco_checkpoints, exist_ok=True)
        os.makedirs(self.forward_checkpoints, exist_ok=True)

    def get_tts_named_weights(self, name):
        """Gets the path for the weights in a named tts checkpoint."""
        return self.taco_checkpoints / f'{name}_weights.pyt'

    def get_tts_named_optim(self, name):
        """Gets the path for the optimizer state in a named tts checkpoint."""
        return self.taco_checkpoints / f'{name}_optim.pyt'

    def get_voc_named_weights(self, name):
        """Gets the path for the weights in a named voc checkpoint."""
        return self.voc_checkpoints/f'{name}_weights.pyt'

    def get_voc_named_optim(self, name):
        """Gets the path for the optimizer state in a named voc checkpoint."""
        return self.voc_checkpoints/f'{name}_optim.pyt'


