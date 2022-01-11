import numpy as np


from ._select_parents_from_generation import _select_parents_from_generation
from ._mate_parents import _mate_parents
from ._Solution import Solution

def create_next_generation(gener, **params):

	population_size = params.get("population_size")
	n_elites = params.get("n_elites")

	new_gener = []

	for i in range(n_elites):
		elite = Solution(solution = gener[i].solution,
							value = gener[i].value,
							flag = 'Elite')
		new_gener.append(elite)

	
	while len(new_gener) < population_size:

		parent1, parent2 = _select_parents_from_generation(gener, **params)
		new_solution1, new_solution2 = _mate_parents(parent1.solution,
													parent2.solution,
													**params)

		for solut in [new_solution1, new_solution2]:
			offspring = Solution(solution = np.copy(solut),
									flag = 'offspring')

			new_gener.append(offspring)

	return new_gener[:population_size]


	