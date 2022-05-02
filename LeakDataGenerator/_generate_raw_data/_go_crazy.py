from .._head_measure import head_measure
from ._generate_random_features import generate_random_features

def _go_crazy(q_out, N, max_n_leaks, L):
	for i in range(N):
		features = generate_random_features(max_n_leaks = max_n_leaks, L = L)
		q_out.put(head_measure(**features))