import torch, torchvision, argparse, os
import numpy as np
import pickle

# import tensorflow as tf
if __name__ == "__main__":
	pytorch_model_path = '/home/kartik/Documents/soft-filter-pruning/small/resnet18/small_model.pt'
	state_dict = torch.load(pytorch_model_path)
	temp = state_dict.state_dict()

	with open('test.pickle', 'wb') as handle:
		pickle.dump(temp, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	# state_dict_keys = state_dict['state_dict']
	for k, v in temp.items():
		print(k, v.shape)