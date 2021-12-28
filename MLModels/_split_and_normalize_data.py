import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from ._get_data import get_data

def _normalize_row(df):

	columns = df.columns
	df = normalize(df, norm = 'l1', axis = 0)
	df = pd.DataFrame(df, columns = columns)

	return df



def _concat(df):
	df_real = df.iloc[:,:25]
	df_imag = df.iloc[:,25:]

	df = pd.concat([_normalize_row(df_real), _normalize_row(df_imag)], axis=1)

	return(df)



def split_and_normalize_data(*args, **kwargs):
	
	X, Y = get_data(**kwargs)
	split_size = kwargs.get('split_size')
	random_state = kwargs.get('random_state')

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split_size, 
															  shuffle = True, 
														      random_state = random_state)

	X_train = _concat(_normalize_row(X_train))
	X_test = _concat(_normalize_row(X_test))

	# scaler = StandardScaler()
	# X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
	# X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns)
	
	return X_train, X_test, Y_train, Y_test	
