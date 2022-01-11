from utils import _load_all_offline_data
from utils import split_and_normalize_data
from utils import evaluate_regression

def train_leak_size(**params):

	split_size = params.get('split_size')
	log = params.get("log")
	model = params.get('model')
	verbose = params.get('verbose')
	direc = params.get("report_directory")

	X, Y, info = _load_all_offline_data(**params)
		
	X_train, X_test, Y_train, Y_test, info_train, info_test = \
		split_and_normalize_data(X, Y,
								info,
								should_normalize = False,
								**params)
	dates_train = X_train.index
	dates_test = X_test.index

	Y_train = Y_train.iloc[:, 0]
	Y_test = Y_test.iloc[:, 0]

	if verbose:
		print ("Trying to fit to the data...")

	model.fit(X_train, Y_train)

	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	evaluate_regression(
		[f'OnTrain-LeakSize', X_train, Y_train, dates_train, info_train],
		[f'OnTest-LeakSize', X_test, Y_test, dates_test, info_test],
		model = model,
		model_name = f"CatBoost",
		logger = log,
		slicer = 1,
		direc = direc)