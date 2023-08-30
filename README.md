# ⏩ ForwardTacotron

Inspired by Microsoft's [FastSpeech](https://www.microsoft.com/en-us/research/blog/fastspeech-new-text-to-speech-model-improves-on-speed-accuracy-and-controllability/)
we modified Tacotron (Fork from fatchord's [WaveRNN](https://github.com/fatchord/WaveRNN)) to generate speech in a single forward pass using a duration predictor to align text and generated mel spectrograms. Hence, we call the model ForwardTacotron (see Figure 1).

<p align="center">
  <img src="assets/model.png" width="700" />
</p>
<p align="center">
  <b>Figure 1:</b> Model Architecture.
</p>

The model has following advantages:
- **Robustness:** No repeats and failed attention modes for challenging sentences.
- **Speed:** The generation of a mel spectogram takes about 0.04s on a GeForce RTX 2080.
- **Controllability:** It is possible to control the speed of the generated utterance.
- **Efficiency:** In contrast to FastSpeech and Tacotron, the model of ForwardTacotron
does not use any attention. Hence, the required memory grows linearly with text size, which makes it possible to synthesize large articles at once.


## UPDATE Improved attention mechanism (30.08.2023)
- Faster tacotron attention buildup by adding alignment conditioning based on [one alignment to rule them all](https://arxiv.org/abs/2108.10447) 
- Improved attention translates to improved synth quality.

## 🔈 Samples

[Can be found here.](https://as-ideas.github.io/ForwardTacotron/)

The samples are generated with a model trained on LJSpeech and vocoded with WaveRNN, [MelGAN](https://github.com/seungwonpark/melgan), or [HiFiGAN](https://github.com/jik876/hifi-gan). 
You can try out the latest pretrained model with the following notebook:  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as-ideas/ForwardTacotron/blob/master/notebooks/synthesize.ipynb)

## ⚙️ Installation

Make sure you have:

* Python >= 3.6

Install espeak as phonemizer backend (for macOS use brew):
```
sudo apt-get install espeak
```

Then install the rest with pip:
```
pip install -r requirements.txt
```

## 🚀 Training your own Model (Singlespeaker)

Change the params in the config.yaml according to your needs and follow the steps below:

(1) Download and preprocess the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset:
 ```
python preprocess.py --path /path/to/ljspeech
```
(2) Train Tacotron with:
```
python train_tacotron.py
```
Once the training is finished, the model will automatically extract the alignment features from the dataset. In case you stopped the training early, you 
can use the latest checkpoint to manually run the process with:
```
python train_tacotron.py --force_align
```
(3) Train ForwardTacotron with:
```
python train_forward.py
```
(4) Generate Sentences with Griffin-Lim vocoder:
```
python gen_forward.py --alpha 1 --input_text 'this is whatever you want it to be' griffinlim
```
If you want to use the [MelGAN](https://github.com/seungwonpark/melgan) vocoder, you can produce .mel files with:
```
python gen_forward.py --input_text 'this is whatever you want it to be' melgan
```
If you want to use the [HiFiGAN](https://github.com/jik876/hifi-gan) vocoder, you can produce .npy files with:
```
python gen_forward.py --input_text 'this is whatever you want it to be' hifigan
```
To vocode the resulting .mel or .npy files use the inference.py script from the MelGAN or HiFiGAN repo and point to the model output folder.

For training the model on your own dataset just bring it to the LJSpeech-like format:
```
|- dataset_folder/
|   |- metadata.csv
|   |- wav/
|       |- file1.wav
|       |- ...
```

For languages other than English, change the language and cleaners params in the hparams.py, e.g. for French:
```
language = 'fr'
tts_cleaner_name = 'no_cleaners'
```

____
You can monitor the training processes for Tacotron and ForwardTacotron with 
```
tensorboard --logdir checkpoints
```
Here is what the ForwardTacotron tensorboard looks like:
<p align="center">
  <img src="assets/tensorboard.png" width="700" />
</p>
<p align="center">
  <b>Figure 2:</b> Tensorboard example for training a ForwardTacotron model.
</p>


## Multispeaker Training
Prepare the data in ljspeech format:
```
|- dataset_folder/
|   |- metadata.csv
|   |- wav/
|       |- file1.wav
|       |- ...
```
The metadata.csv is expected to have the speaker id in the second column:
```
id_001|speaker_1|this is the first text.
id_002|speaker_1|this is the second text.
id_003|speaker_2|this is the third text.
...
```
We also support the VCTK and a pandas format 
(can be set in the config multispeaker.yaml under preprocesing.metafile_format)

Follow the same steps as for singlespaker, but provide the multispeaker config:
 ```
python preprocess.py --config configs/multispeaker.yaml --path /path/to/ljspeech
python train_tacotron.py --config configs/multispeaker.yaml
python train_forward.py --config configs/multispeaker.yaml
```

## Pretrained Models

| Model | Dataset | Commit Tag |
|---|---|------------|
|[forward_tacotron](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ForwardTacotron/forward_step90k.pt)| ljspeech | v3.1     |
|[fastpitch](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ForwardTacotron/thorsten_fastpitch_50k.pt)| [thorstenmueller (german)](https://github.com/thorstenMueller/deep-learning-german-tts) | v3.1     |

Our pre-trained LJSpeech model is compatible with the pre-trained vocoders:
- [MelGAN](https://github.com/seungwonpark/melgan)
- [HiFiGAN](https://github.com/jik876/hifi-gan)


After downloading the models you can synthesize text using the pretrained models with
```
python gen_forward.py --input_text 'Hi there!' --checkpoint forward_step90k.pt wavernn --voc_checkpoint wave_step_575k.pt

```

## Export Model with TorchScript

Here is a dummy example of exporting the model in TorchScript:
```
import torch
from models.forward_tacotron import ForwardTacotron

tts_model = ForwardTacotron.from_checkpoint('checkpoints/ljspeech_tts.forward/latest_model.pt')
tts_model.eval()
model_script = torch.jit.script(tts_model)
x = torch.ones((1, 5)).long()
y = model_script.generate_jit(x)
```
For the necessary preprocessing steps (text to tokens) please refer to:
```
gen_forward.py
```

## References

* [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)
* [FastPitch: Parallel Text-to-speech with Pitch Prediction](https://arxiv.org/abs/2006.06873)
* [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
* [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711)
* [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558)

## Acknowlegements

* [https://github.com/keithito/tacotron](https://github.com/keithito/tacotron)
* [https://github.com/fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
* [https://github.com/seungwonpark/melgan](https://github.com/seungwonpark/melgan)
* [https://github.com/jik876/hifi-gan](https://github.com/jik876/hifi-gan)
* [https://github.com/xcmyz/LightSpeech](https://github.com/xcmyz/LightSpeech)
* [https://github.com/resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
* [https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)
* [https://github.com/resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer)

## Maintainers

* Christian Schäfer, github: [cschaefer26](https://github.com/cschaefer26)

## Copyright

See [LICENSE](LICENSE) for details.
