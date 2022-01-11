import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ._normalize_rowwise import _normalize_rowwise

def split_and_normalize_data(X, Y, info, should_normalize, **params):

	split_size = params.get('split_size')
	random_seed = params.get('random_seed')
	input_dim = params.get("input_dim")
	verbose = params.get("verbose")

	if random_seed is None:
		raise ValueError("Take care of random seed carefully")
	if verbose:
		print ("Trying to split and normalize data")

	X_train, X_test, Y_train, Y_test, info_train, info_test = \
			train_test_split(X, Y, info,
							test_size = split_size, 
							shuffle = True,
							random_state = random_seed)

	if should_normalize:
		X_train = _normalize_rowwise(X_train, input_dim)
		X_test = _normalize_rowwise(X_test, input_dim)
	
	return X_train, X_test, Y_train, Y_test, info_train, info_test
