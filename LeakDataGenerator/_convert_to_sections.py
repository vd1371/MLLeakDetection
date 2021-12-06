import numpy as np
import pandas as pd

def convert_to_sections(df, n_sections = 40, L =2000, max_n_leaks = 3):

	l_sections = int(L/n_sections)

	df.loc[:, 'xL0':'xL2'] = ((df.loc[:, 'xL0':'xL2']-0.0000001) / l_sections)

	# Converting to integer
	cols = ['xL0', 'xL1', 'xL2']
	df[cols] = df[cols].applymap(np.int16)

	holder_locs = []
	holder_size = []

	for idx in df.index:

		tmp = list(df.loc[idx, 'X0':'X49'])

		leak_locs = np.zeros(n_sections)
		leak_size = np.zeros(n_sections)

		for i in range(max_n_leaks):

			leak_idx = df.loc[idx, f'xL{i}']

			leak_locs[leak_idx] = 1
			leak_size[leak_idx] = df.loc[idx, f'CdAl{i}']

		holder_locs.append(tmp + leak_locs.tolist())
		holder_size.append(tmp + leak_size.tolist())

	# Saving the Leak Locs and Leak Sizes
	cols = [f'X{i}' for i in range(50)] + [f'LL{j}' for j in range(n_sections)]
	df_leak_locs = pd.DataFrame(holder_locs, columns = cols)

	cols = [f'X{i}' for i in range(50)] + [f'LS{j}' for j in range(n_sections)]
	df_leak_size = pd.DataFrame(holder_size, columns = cols)

	return df_leak_locs, df_leak_size
