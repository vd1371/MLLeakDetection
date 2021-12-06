# Loading dependencies
import numpy as np

def generate_random_features(random_location = True,
							random_size = True):

	# pipe length. range 100:10'000
	L = 2000
	# sound speed. 900:1200 important
	a = 1200
	# pipe diameter  not important        
	D = 0.5
	# area of pipe
	A = np.pi*(D/2)**2
	# friction parameter
	f = 0.00
	n_leaks = 3
	max_n_leaks = 3

	# location of leaks: two leaks far away from each other
	if random_location:
		# n_leaks = np.random.choice([1, 2, 3])
		xL = list(np.array(sorted(np.random.random(max_n_leaks)))*L)
	else:
		xL = list(np.array([0.8, 0.6, 0.25])*L)

	if random_size:
		CdAl = np.random.random(max_n_leaks)
	else:
		CdAl = np.array([0.01, 0.01, 0.01])

	return {"L": L,
		'n_leaks': n_leaks,
		'max_n_leaks': max_n_leaks,
		'xL': xL,
		'sound_speed': a,
		"diameter": D,
		'Area_of_pipe': A,
		'friction_parameter': f,
		'CdAl': CdAl
	}
