import os
import argparse
import torch
from models.forward_tacotron import ForwardTacotron


"""
Torchscript exporter for ⏩ ForwardTacotron
"""


# Declaring the convertor:
def run_convertor(model_path, save_path):
	if not os.path.exists(model_path):
		raise FileNotFoundError("Please give me an existing model!")
	tts_model = ForwardTacotron.from_checkpoint(model_path)
	tts_model.eval()
	# Initialize a defined TTS model for torchscript in models/ForwardTacotron:
	model_script = torch.jit.script(tts_model)
	# Generate input for testing:
	x = torch.ones((1, 5)).long()
	# Try generating this input:
	y = model_script.generate_jit(x)
	if save_path is None:
		save_path = model_path[:-3]+".ts"
	# Finally, we export it:
	torch.jit.save(
		model_script,
		save_path
	)
	print("Model successfully converted to torchscript.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="TorchScript convertor for ⏩ForwardTacotron")
	parser.add_argument(
		'--checkpoint_path',
		'-c',
		required=True,
		type=str,
		help='The full checkpoint (*.pt) file to convert.'
	)
	parser.add_argument(
		'--output_path',
		'-o',
		default=None,
		type=str,
		help='Output path to save the converted TorchScript model.'
	)
	args = parser.parse_args()
	run_convertor(args.checkpoint_path, args.output_path)