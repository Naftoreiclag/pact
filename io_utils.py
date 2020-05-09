import numpy as np
import json
import os
import io
import base64
from PIL import Image

def json_save(json_data, fname):
	with open(fname, 'w') as ofile:
		json.dump(json_data, ofile)
		
def json_load(fname):
	with open(fname, 'r') as ifile:
		json_data = json.load(ifile)
		return json_data
		
def save_matrix_to_json(mat):
	return mat.tolist()

def load_matrix_from_json(json_data):
	return np.array(json_data)
	
def save_image_to_string(np_image):
	in_memory_png_file = io.BytesIO()
	save_raw_image(in_memory_png_file, np_image)
	return base64.b64encode(in_memory_png_file.getvalue()).decode('ascii') 

def load_image_from_string(string):
	return load_raw_image(io.BytesIO(base64.b64decode(string)))

def load_raw_image(file_or_fname):
	img = Image.open(file_or_fname)
	return np.asarray(img, dtype=np.float32) / 255

def save_raw_image(file_or_fname, numpy_array):
	img = Image.fromarray((numpy_array * 255).astype(np.uint8))
	img.save(file_or_fname, format='PNG')

def try_convert_to_relative_path(fname):
	cwd = os.getcwd()
	abs_fname = os.path.realpath(fname)
	abs_cwd = os.path.realpath(cwd)
	common = os.path.relpath(abs_fname, abs_cwd)
	return common

