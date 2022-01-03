from ._get_data import get_data
from ._get_call_backs import _get_call_backs
from ._save_model import _save_model

from utils import split_and_normalize_data
from utils import evaluate_classification

from utils import Logger

def TrainLeakLocs(**kwargs):
	
	warm_up = kwargs.get('warm_up')
	starting_batch = kwargs.get('starting_batch')
	n_rounds = kwargs.get('n_rounds')
	split_size = kwargs.get('split_size')
	epochs = kwargs.get('epochs')
	batch_size = kwargs.get('batch_size')
	log = kwargs.get("log")
	model = kwargs.get('model')
	verbose = kwargs.get('verbose')	

	call_back_list = _get_call_backs()

	for batch_number in range(starting_batch, n_rounds):
		
		X, Y = get_data(batch_number, **kwargs)
		
		X_train, X_test, Y_train, Y_test = split_and_normalize_data(X, Y)

		if verbose:
			print ("Trying to fit to the new generated data...")
		
		model.fit(X_train, Y_train,
				  validation_split=split_size,
				  epochs=epochs,
				  batch_size=batch_size,
			      verbose = 2, 
			      shuffle=True, 
				  callbacks=call_back_list)

		# Evaluate the model
		train_scores = model.evaluate(X_train, Y_train, verbose=2)
		test_scores = model.evaluate(X_test, Y_test, verbose=2)
		
		if verbose:
			print (f'Trian_err: {train_scores}, Test_err: {test_scores}')
		log.info(f'batch_number:{batch_number}, Trian_err: {train_scores}, Test_err: {test_scores}')


		raise ValueError ("Please take care of save model at your convenience")
		_save_model()

		y_pred_train = model.predict(X_train)
		y_pred_test = model.predict(X_test)

		evaluate_classification([f'OnTrain-xL{i}', X_train, Y_train, dates_train],
								[f'OnTest-xL{i}', X_test, Y_test, dates_test],
								model = model,
								model_name = f"DNN",
								logger = log,
								slicer = 1)