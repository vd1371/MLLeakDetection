# Loading dependencies
import numpy as np

def generate_random_features(random_location = True,
							random_size = True,
							**params):

	max_n_leaks = params.get("max_n_leaks")
	L = params.get("L")

	# location of leaks: two leaks far away from each other
	if random_location:
		xL = list(np.array(sorted(np.random.random(max_n_leaks)))*L)
	else:
		xL = list(np.array([0.1, 0.3, 0.8])*L)

	if random_size:
		CdAl = np.random.exponential(0.2, max_n_leaks)
		# CdAl = np.where(CdAl < 0.1, 0.1, CdAl)
		CdAl = np.where(CdAl > 1, 1, CdAl)
	else:
		CdAl = np.array([0.01, 0.01, 0.01])

	return {
		'xL': xL,
		'CdAl': CdAl,
		'L': L,
	}
