from ._optimize_linear import _optimize_linear
from ._optimize_parallel import _optimize_parallel
from ._save_solutions_to_csv import save_solutions_to_csv
from ._report_accuracy_of_solutions import report_accuracy_of_solutions

def optimize(data, **params):
		
	n_cores = params.get("n_cores")

	if n_cores == 1:
		holder_of_solutions_and_info = _optimize_linear(data, **params)
	else:
		holder_of_solutions_and_info = _optimize_parallel(data, **params)

	df = save_solutions_to_csv(holder_of_solutions_and_info, **params)
	report_accuracy_of_solutions(df, **params)

	print ("Optimization done")