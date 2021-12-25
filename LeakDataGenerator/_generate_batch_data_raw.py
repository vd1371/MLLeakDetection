from ._generate_batch_data_raw_linear import _generate_batch_data_raw_linear
from ._generate_batch_data_raw_parallel import _generate_batch_data_raw_parallel

def _generate_batch_data_raw(**params):
	batch_size_of_generator = params.get("batch_size_of_generator")
	verbose = params.get("verbose")
	n_cores = params.get("n_cores")
	# holder = []

	if n_cores == 1:		
		return _generate_batch_data_raw_linear(**params)
	else:
		return _generate_batch_data_raw_parallel(**params)
