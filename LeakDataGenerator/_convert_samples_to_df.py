import numpy as np
import pandas as pd

def convert_samples_to_df(samples):

	# Creating colomn names
	first_sample = samples[0][0]
	cols = []
	for k in first_sample.keys():
		if k == "xL":
			cols += [f"xL{i}" for i in range(first_sample['max_n_leaks'])]
		elif k == "CdAl":
			cols += [f"CdAl{i}" for i in range(first_sample['max_n_leaks'])]
		else:
			cols.append(k)
	cols = cols + [f'X{i}' for i in range(len(samples[0][1])*2)]


	'''Converting the data from holder to CSV'''
	holder = []
	for params, results in samples:

		# Adding the pipe and leakages parameters
		tmp_holder = []
		for k, v in params.items():

			if k in ['xL', 'CdAl']:
				tmp_holder += list(v)
			else:
				tmp_holder.append(v)

		for real, imag in zip(np.real(results), np.imag(results)):

			tmp_holder.append(real)
			tmp_holder.append(imag)

		holder.append(tmp_holder)

	indices = [generate_indices() for _ in range(len(holder))]
	df = pd.DataFrame(holder, columns = cols, index = indices)


	return df

def generate_indices():

	ind = int(np.random.random()*np.random.random()*np.random.random()*1000000000)
	return ind