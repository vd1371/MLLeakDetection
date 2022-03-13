from .._head_measure import head_measure
from ._generate_random_features import generate_random_features

def _go_crazy(q_out, N = 10000):
	for i in range(N):
		features = generate_random_features()
		q_out.put(h_d_measure(**features))