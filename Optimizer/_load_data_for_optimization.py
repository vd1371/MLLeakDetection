from utils import _load_all_offline_data
from utils import split_and_normalize_data

def _load_data_for_optimization(**params):

	n_samples = params.get("n_samples")

	X, Y, info = _load_all_offline_data(**params)
	_, X_test, _, Y_test, _, info_test = \
		split_and_normalize_data(X, Y,
								info,
								should_normalize = False,
								**params)

	X_test = X_test.iloc[:n_samples, :]
	Y_test = Y_test.iloc[:n_samples, :]
	info_test = info_test.iloc[:n_samples, :]

	return X_test, Y_test, info_test





