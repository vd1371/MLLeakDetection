from ._get_data import get_data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_and_normalize_data(*args, **kwargs):
	
	X, Y = get_data()
	split_size = kwargs.get('split_size')
	random_state = kwargs.get('random_state')

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split_size, 
															  shuffle = True, 
														      random_state = random_state)

	scaler = StandardScaler()
	X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
	X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns)
	
	return X_train, X_test, Y_train, Y_test	