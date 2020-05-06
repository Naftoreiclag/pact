import numpy as np
import json

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
