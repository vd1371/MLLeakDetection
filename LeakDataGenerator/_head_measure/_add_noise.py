import numpy as np

from ._head_measure_exact_model import _head_measure_exact_model

def _add_noise(hd_ff, **params):

	max_n_leaks = params.get("max_n_leaks")
	L = params.get("L")
	sigma = params.pop("sigma_noise")

	noise_vector = 1 + np.random.normal(0, sigma, len(hd_ff))
	hd_ff_real = np.multiply(hd_ff.real, noise_vector)

	noise_vector = 1 + np.random.normal(0, sigma, len(hd_ff))
	hd_ff_imag = np.multiply(hd_ff.imag, noise_vector)

	return hd_ff_real + hd_ff_imag * 1j