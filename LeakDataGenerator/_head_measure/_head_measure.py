# Loading dependencies
import numpy as np

from ._head_measure_exact_model import _head_measure_exact_model
from ._add_noise import _add_noise

def head_measure(**params):

	xL = params.get('xL')
	CdAl = params.get('CdAl')
	L = params.get("L")
	max_n_leaks = params.get("max_n_leaks")
	sigma_noise = params.get("sigma_noise")

	hd_ff = _head_measure_exact_model(**params)
	if sigma_noise > 0:
		hd_ff = _add_noise(hd_ff, **params)

	return (params, hd_ff)