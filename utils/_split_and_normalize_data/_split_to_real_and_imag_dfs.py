import numpy as np

def _split_to_real_and_imag_dfs(df, input_dim):

	even_indices = np.arange(0, input_dim, 2, dtype=int)
	odd_indices = np.arange(1, input_dim, 2, dtype=int)

	df_real = df.iloc[:,even_indices]
	df_imag = df.iloc[:,odd_indices]

	return df_real, df_imag