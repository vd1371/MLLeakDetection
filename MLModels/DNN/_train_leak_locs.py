from ._get_call_backs import _get_call_backs
from ._save_model import _save_model

from utils import split_and_normalize_data
from utils import evaluate_classification
from utils import _load_all_offline_data

def train_leak_locs(**params):

	log = params.get("log")	
	model = params.get('model')
	model_name = params.get("model_name")
	verbose = params.get('verbose')	
	report_directory = params.get('report_directory')
	split_size = params.get('split_size')
	epochs = params.get('epochs')
	batch_size = params.get('batch_size')

	call_back_list = _get_call_backs()
		
	X, Y, info = _load_all_offline_data(**params)
		
	X_train, X_test, Y_train, Y_test, info_train, info_test = \
		split_and_normalize_data(X, Y,
								info,
								should_normalize = False,
								**params)
	dates_train = X_train.index
	dates_test = X_test.index

	Y_train = Y_train.iloc[:, 1]
	Y_test = Y_test.iloc[:, 1]

	if verbose:
		print ("Trying to fit to the data...")

	# print(X_train.shape)
	# print('-------------------------')
	# print(Y_train.shape)
	# raise ValueError
		
	model.fit(X_train, Y_train,
			  validation_split=split_size,
			  epochs=epochs,
			  batch_size=batch_size,
		      verbose = 2, 
		      shuffle=True, 
			  callbacks=call_back_list)

	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	evaluate_classification(
			[f'OnTrain-LeakLocs', X_train, Y_train, dates_train, y_pred_train, info_train],
			[f'OnTest-LeakLocs', X_test, Y_test, dates_test, y_pred_test, info_test],
			model_name = model_name,
			logger = log,
			report_directory = report_directory)