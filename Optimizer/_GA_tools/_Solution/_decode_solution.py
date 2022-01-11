import numpy as np
from ._binary_to_num import _binary_to_num

def _decode_solution(encoded_form, **params):

	location_precision = params.get("location_precision")
	location_binary_length = params.get("location_binary_length")
	size_precision = params.get("size_precision")
	size_binary_length = params.get("size_binary_length")
	max_n_leaks = params.get("max_n_leaks")

	xL, CdAl = [], []
	for i in range(max_n_leaks):

		leak_binary = encoded_form[i*location_binary_length: \
								(i+1)*location_binary_length]
		leak_loc = _binary_to_num(leak_binary, location_precision)
		xL.append(leak_loc)

		size_binary = encoded_form[(max_n_leaks+i)*size_binary_length: \
								(max_n_leaks+i+1)*size_binary_length]
		leak_size = _binary_to_num(size_binary, size_precision)
		CdAl.append(leak_size)

	return {'xL': sorted(xL),
			'CdAl': np.array(CdAl)}

