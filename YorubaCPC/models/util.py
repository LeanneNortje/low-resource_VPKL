#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import json
import torch
import os
from os import popen
from math import ceil
from models.ImageModels import *
from models.AudioModels import *

terminal_rows, terminal_width = popen('stty size', 'r').read().split()
terminal_width = int(terminal_width)

image_model_dict = {
	"VGG16": VGG16,
	"Resnet50": Resnet50
}
audio_model_dict = {
	"Davenet": AudioCNN,
	"ResDavenet": ResidualAudioCNN
}

def heading(string):
    print("_"*terminal_width + "\n")
    print(string)
    print("_"*terminal_width + "\n")

def imageModel(args):
	if args["image_model"] == "VGG16":
		return image_model_dict["VGG16"]
	elif args["image_model"] == "Resnet50":
		return image_model_dict["Resnet50"]
	else:
		raise ValueError(f'Unknown image model: {args["image_model"]["name"]}')

def audioModel(args):
	if args["audio_model"]["name"] == "DAVEnet":
		with open(f'models/DAVEnet.json') as file: model_params = json.load(file)
		args["audio_model"]["conv_layers"] = model_params["conv_layers"]
		args["audio_model"]["max_pool"] = model_params["max_pool"]
		return audio_model_dict["Davenet"]
	elif args["audio_model"]["name"] == "ResDAVEnet":
		with open(f'models/ResDAVEnet.json') as file: model_params = json.load(file)
		args["audio_model"]["conv_layers"] = model_params["conv_layers"]
		return audio_model_dict["ResDavenet"]
	else:
		raise ValueError(f'Unknown audio model: {args["audio_model"]["name"]}')

def acousticModel(args):
	with open(f'models/AcousticEncoder.json') as file: model_params = json.load(file)
	args["acoustic_model"] = model_params
	
