import os
from pathlib import Path


class Paths:
    """Manages and configures the paths used by WaveRNN, Tacotron, and the data."""
    def __init__(self, data_path, voc_id, tts_id):
        self.base = Path(__file__).parent.parent.expanduser().resolve()

        # Data Paths
        self.data = Path(data_path).expanduser().resolve()
        self.quant = self.data/'quant'
        self.mel = self.data/'mel'
        self.gta = self.data/'gta'
        self.att_pred = self.data/'att_pred'
        self.alg = self.data/'alg'
        self.raw_pitch = self.data/'raw_pitch'
        self.phon_pitch = self.data/'phon_pitch'
        self.phon_energy = self.data/'phon_energy'

        self.model_output = self.base / 'model_output'

        self.voc_checkpoints = self.base/'checkpoints'/f'{voc_id}.wavernn'
        self.voc_top_k = self.voc_checkpoints/'top_k_models'
        self.voc_log = self.voc_checkpoints/'logs'

        self.taco_checkpoints = self.base / 'checkpoints' / f'{tts_id}.tacotron'
        self.taco_log = self.taco_checkpoints / 'logs'

        self.forward_checkpoints = self.base/'checkpoints'/f'{tts_id}.forward'
        self.forward_log = self.forward_checkpoints/'logs'

        self.create_paths()

    def create_paths(self):
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.quant, exist_ok=True)
        os.makedirs(self.mel, exist_ok=True)
        os.makedirs(self.gta, exist_ok=True)
        os.makedirs(self.alg, exist_ok=True)
        os.makedirs(self.att_pred, exist_ok=True)
        os.makedirs(self.raw_pitch, exist_ok=True)
        os.makedirs(self.phon_pitch, exist_ok=True)
        os.makedirs(self.phon_energy, exist_ok=True)
        os.makedirs(self.voc_checkpoints, exist_ok=True)
        os.makedirs(self.voc_top_k, exist_ok=True)
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


