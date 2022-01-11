import numpy as np
from deap import tools

def _mate_parents(solut1, solut2, **params):

	crossver_prob = params.get("crossver_prob")
	mutation_prob = params.get("mutation_prob")

	if np.random.random() < crossver_prob:
		solut1, solut2 = tools.cxTwoPoint(solut1, solut2)

	solut1 = tools.mutFlipBit(solut1, mutation_prob)[0]
	solut2 = tools.mutFlipBit(solut2, mutation_prob)[0]

	return solut1, solut2