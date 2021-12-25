from ._h_d_measure import _h_d_measure
from ._generate_random_features import generate_random_features

def _go_crazy(q_out, N = 10000):
	for i in range(N):
		features = generate_random_features()
		q_out.put(_h_d_measure(**features))

'''
def _go_crazy(**params):
	N = params.get("N")
	q_out = params.get("q_out")

	for i in range(N):
		features = generate_random_features()
		q_out.put(h_d_measure(**features))
'''