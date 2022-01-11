
from ._cost_function import _cost_function


def evaluate_generation(gener, n_gener, sample_signal, **params):

	n_elites = params.get("n_elites")

	idx = n_elites if n_gener != 0 else 0
	elites, to_be_eval = gener[:idx], gener[idx:]

	for i, soloution in enumerate(to_be_eval):
		
		cost = _cost_function(soloution, sample_signal, **params)
		soloution.set_value(cost)
		
	gener = to_be_eval if idx == 0 else elites + to_be_eval
	gener = sorted(gener, key=lambda x: x.value, reverse = False)

	return gener