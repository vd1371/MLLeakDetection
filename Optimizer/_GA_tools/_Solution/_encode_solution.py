from ._num_to_binary import _num_to_binary

def _encode_solution(original_form, **params):

	location_precision = params.get("location_precision")
	location_binary_length = params.get("location_binary_length")
	size_precision = params.get("size_precision")
	size_binary_length = params.get("size_binary_length")
	max_n_leaks = params.get("max_n_leaks")

	encoded_form = []

	for i, val in enumerate(original_form):
		if i < max_n_leaks:
			tmp = _num_to_binary(val,
								location_binary_length,
								location_precision)

		else:
			tmp = _num_to_binary(val,
								size_binary_length,
								size_precision)

		encoded_form += tmp

	return encoded_form