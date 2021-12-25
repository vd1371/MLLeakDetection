from ._generate_random_features import generate_random_features
from ._h_d_measure import _h_d_measure

def _generate_batch_data_raw_linear(**params):
	batch_size_of_generator = params.get('batch_size_of_generator')
	verbose = params.get('verbose')

	for i in range(batch_size_of_generator):

		random_features = generate_random_features()
		h_d_data = h_d_measure(random_features)
		holder.append(h_d_data)

		if i % 1000 == 0 and verbose:
			print (f"{i} samples are generated now")
	
	return holder