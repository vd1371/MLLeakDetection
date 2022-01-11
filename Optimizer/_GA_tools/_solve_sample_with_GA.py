from ._initialize_generation import initialize_generation
from ._create_next_generation import create_next_generation
from ._evaluate_generation import evaluate_generation


def solve_sample_with_GA(sample_leak_info,
						sample_signal,
						**params):

	n_generations = params.get("n_generations")
	crossver_prob = params.get("crossver_prob")
	mutation_prob = params.get("mutation_prob")
	population_size = params.get("population_size")
	n_elites = params.get("n_elites")

	n_gener = 0
	best_values, gener_num_holder = [], []

	first = True
	for n_gener in range(n_generations):

		if first:
			gener = initialize_generation(**params)
			first = False
		else:
			gener = create_next_generation(gener, **params)

		gener = evaluate_generation(gener,
									n_gener,
									sample_signal,
									**params)

		# print (n_gener, round(gener[0].value, 1))

	return gener[0].decode(**params)
