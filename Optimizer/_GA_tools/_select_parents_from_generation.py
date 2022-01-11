import numpy as np

def _select_parents_from_generation(gener, **params):

	probs = params.get("probs")

	parent1, parent2 = np.random.choice(gener,
										size = (2,),
										p= probs,
										replace = False)

	return parent1, parent2