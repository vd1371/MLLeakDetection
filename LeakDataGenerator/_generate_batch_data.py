
from ._generate_random_features import generate_random_features
from ._h_d_measure import h_d_measure

def generate_batch_data(N = 100, verbose = True):

	holder = []
	for i in range(N):
		if i % 1000 == 0 and verbose:
			print (f"{i} samples are generated now")

		random_features = generate_random_features()
		h_d_data = h_d_measure(random_features)

		holder.append(h_d_data)

	return holder