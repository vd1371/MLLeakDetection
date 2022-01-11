import numpy as np

from numpy.linalg import norm

from LeakDataGenerator import head_measure

def _cost_function(soloution, sample_signal, **params):

	leak_info = soloution.decode(**params)
	max_n_leaks = params.get("max_n_leaks")
	L = params.get("L")

	for i in range(max_n_leaks):
		if leak_info['xL'][i] > L or \
			leak_info['CdAl'][i] > 1:
			return np.inf

	individual_signal = head_measure(**{**leak_info,
									**params})[1]
	diff = norm((individual_signal - sample_signal)**2)

	return diff