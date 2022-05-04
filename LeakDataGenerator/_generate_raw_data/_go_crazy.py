from .._head_measure import head_measure
from ._generate_random_features import generate_random_features

def _go_crazy(q_out, N, max_n_leaks, L, max_omeg_num, sigma_noise):
	holder = []

	for i in range(N):
		features = generate_random_features(max_n_leaks = max_n_leaks, L = L)
		h_d_data = head_measure(**{**features,
									**{'max_n_leaks': max_n_leaks,
									'max_omeg_num': max_omeg_num,
									'sigma_noise': sigma_noise}})
		holder.append(h_d_data)

	q_out.put(holder)