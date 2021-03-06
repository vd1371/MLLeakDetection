import numpy as np
import pandas as pd

def convert_to_sections(df, **params):

	n_sections = params.get("n_sections")
	max_n_leaks = params.get("max_n_leaks")
	L = params.get("L")
	max_omeg_num = params.get("max_omeg_num")

	l_sections = int(L/n_sections)

	for i in range(max_n_leaks):
		df[f'xind{i}'] = (df.loc[:, f'xL{i}'].values-0.0000001)/l_sections

	# Converting to integer
	cols = ['xind0', 'xind1', 'xind2']
	df[cols] = df[cols].applymap(np.int16)

	holder_locs = []
	holder_size = []

	for idx in df.index:

		tmp = list(df.loc[idx, 'X0':f'X{2*max_omeg_num-1}'])

		leak_locs = np.zeros(n_sections)
		leak_size = np.zeros(n_sections)

		for i in range(max_n_leaks):

			leak_idx = df.loc[idx, f'xind{i}']

			leak_locs[int(leak_idx)] = 1
			leak_size[int(leak_idx)] = df.loc[idx, f'CdAl{i}']

		holder_locs.append(tmp + leak_locs.tolist())
		holder_size.append(tmp + leak_size.tolist())

	# Saving the Leak Locs and Leak Sizes
	cols = [f'X{i}' for i in range(max_omeg_num*2)] + \
				[f'LL{j}' for j in range(n_sections)]
	df_leak_locs = pd.DataFrame(holder_locs, columns = cols)

	cols = [f'X{i}' for i in range(max_omeg_num*2)] + \
				[f'LS{j}' for j in range(n_sections)]
	df_leak_size = pd.DataFrame(holder_size, columns = cols)

	leak_info_cols = ['xL0', 'xL1', 'xL2', 'CdAl0', 'CdAl1', 'CdAl2']
	for col in leak_info_cols:
		df_leak_locs[col] = df[col].values
		df_leak_size[col] = df[col].values

	return df_leak_locs, df_leak_size
