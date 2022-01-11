import numpy as np

def _convert_df_to_signals(signals_df, **params):

	input_dim = params.get("input_dim")

	even_indices = np.arange(0, input_dim, 2, dtype=int)
	odd_indices = np.arange(1, input_dim, 2, dtype=int)
	df_real = signals_df.iloc[:,even_indices]
	df_imag = signals_df.iloc[:,odd_indices]
	
	signals = df_real.values + df_imag.values * 1j

	return signals