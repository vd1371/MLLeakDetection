import numpy as np

from ._Solution import Solution

def initialize_generation(**params):

	pop_size = params.get("population_size")
	location_binary_length = params.get("location_binary_length")
	size_binary_length = params.get("size_binary_length")
	max_n_leaks = params.get("max_n_leaks")

	solut = np.random.randint(2, 
				size = max_n_leaks *(location_binary_length + \
										size_binary_length))

	gener = [Solution(solution = solut,
						flag = "FirstGener" ,
						**params) for _ in range(pop_size)]
	
	return gener
