

def _optimize_parallel(data, **params):
	'''
	Tips:
	Seperate data into smaller pieces for each core
	define each process as one _optimize_linear
	pass the smaller data to the _optimize_linear function
	collect all the holders and combine them
	'''
	raise NotImplementedError("Kept for MHK")