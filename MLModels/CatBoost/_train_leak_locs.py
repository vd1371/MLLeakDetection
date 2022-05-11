import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from utils import _load_all_offline_data
from utils import split_and_normalize_data
from utils import evaluate_classification

def train_leak_locs(**params):
	
	log = params.get("log")
	model = params.get('model')
	model_name = params.get("model_name")
	verbose = params.get('verbose')
	report_directory = params.get('report_directory')
	noises = params.get("noises")

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

	model.fit(X_train, Y_train)

	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	evaluate_classification(
			[f'OnTrain-LeakLocs', X_train, Y_train, dates_train, y_pred_train, info_train],
			[f'OnTest-LeakLocs', X_test, Y_test, dates_test, y_pred_test, info_test],
			model_name = model_name,
			logger = log,
			report_directory = report_directory)

	if len(noises) > 1:
	    df_noise = pd.DataFrame(columns=['noise','acc','f1'])
	    for i, noise in enumerate(noises):
	        df_noise.loc[i,'noise'] = noise

	    for i, noise in enumerate(noises):
	    	noise_vector = np.random.normal(0, noise, (X_test.shape[0], X_test.shape[1]))
	    	X_test_noise = pd.DataFrame(np.add(np.array(X_test), noise_vector), columns = X_test.columns)
	    	y_pred_test_noise = model.predict(X_test_noise)

	    	df_noise.loc[i,'noise'] = noise
	    	df_noise.loc[i,'acc'], df_noise.loc[i,'f1'] = \
	    		accuracy_score(Y_test, y_pred_test_noise)*100, f1_score(Y_test, y_pred_test_noise)*100

	    	evaluate_classification(
				[f'OnTrain-LeakLocs', X_train, Y_train, dates_train, y_pred_train, info_train],
				[f'OnTest-LeakLocs-noise-{noise}', X_test_noise, Y_test, dates_test, y_pred_test_noise, info_test],
				model_name = model_name,
				logger = log,
				report_directory = report_directory)

	    	del noise_vector, X_test_noise, y_pred_test_noise

	    df_noise.to_csv(report_directory + "/" + f'{model_name}-LeakLocs-noise.csv', index=False)





	# for noise in range(0, 1, 2, 5, 10, 15, 20):

		# ADD noise to X_test
		# Find the metrics of the noisy test set

	# Save the report as a csv file
