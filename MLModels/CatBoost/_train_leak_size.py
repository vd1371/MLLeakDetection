from utils import _load_all_offline_data
from utils import split_and_normalize_data
from utils import evaluate_regression

def train_leak_size(**params):

	log = params.get("log")
	model = params.get('model')
	model_name = params.get("model_name")
	verbose = params.get('verbose')
	report_directory = params.get('report_directory')
	leak_pred = params.get("leak_pred")

	X, Y, info = _load_all_offline_data(**params)
	
	section_number = 1

	if leak_pred == "LeakSize":
		mask = Y.iloc[:, section_number] > 0
		X = X.loc[mask, :]
		Y = Y.loc[mask, :]
		info = info.loc[mask, :]

	X_train, X_test, Y_train, Y_test, info_train, info_test = \
		split_and_normalize_data(X, Y,
								info,
								should_normalize = False,
								**params)
	dates_train = X_train.index
	dates_test = X_test.index

	Y_train = Y_train.iloc[:, section_number]
	Y_test = Y_test.iloc[:, section_number]

	if verbose:
		print ("Trying to fit to the data...")

	model.fit(X_train, Y_train)

	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	evaluate_regression(
		[f'OnTrain-LeakSize', X_train, Y_train, dates_train, y_pred_train, info_train],
		[f'OnTest-LeakSize', X_test, Y_test, dates_test, y_pred_test, info_test],
		model = model,
		model_name = model_name,
		logger = log,
		report_directory = report_directory)