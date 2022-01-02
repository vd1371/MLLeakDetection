import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def _normalize_row(df):

	columns = df.columns
	df = normalize(df, norm = 'l1', axis = 0)
	df = pd.DataFrame(df, columns = columns)

	return df

def _concat(df, input_dim):

	even_indices = np.arange(0, input_dim, 2, dtype=int)
	odd_indices = np.arange(1, input_dim, 2, dtype=int)

	df_real = df.iloc[:,even_indices]
	df_imag = df.iloc[:,odd_indices]

	df = pd.concat([_normalize_row(df_real), _normalize_row(df_imag)], axis=1)

	return df

def split_and_normalize_data(X, Y, **params):

	split_size = params.get('split_size')
	random_state = params.get('random_state')
	input_dim = params.get("input_dim")
	verbose = params.get("verbose")

	if verbose:
		print ("Trying to split and normalize data")

	X_train, X_test, Y_train, Y_test = \
			train_test_split(X, Y,
							test_size = split_size, 
							shuffle = True,
							random_state = random_state)

	X_train = _concat(_normalize_row(X_train), input_dim)
	X_test = _concat(_normalize_row(X_test), input_dim)
	
	return X_train, X_test, Y_train, Y_test
