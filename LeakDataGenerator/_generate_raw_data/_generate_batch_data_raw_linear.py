from ._generate_random_features import generate_random_features
from .._head_measure import head_measure

def _generate_batch_data_raw_linear(**params):
	batch_size_of_generator = params.get('batch_size_of_generator')
	verbose = params.get('verbose')

	holder = []
	for i in range(batch_size_of_generator):

		random_features = generate_random_features(**params)
		h_d_data = head_measure(**{**random_features,
									**params})
		holder.append(h_d_data)

		# if i % 1000 == 0 and verbose:
		# 	print (f"{i} samples are generated now")
	
	return holder