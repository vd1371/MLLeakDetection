import pandas as pd

from ._split_to_real_and_imag_dfs import _split_to_real_and_imag_dfs
from ._normalize_df_parallel import _normalize_df_parallel

def _normalize_rowwise(df, input_dim):

	df_real, df_imag = _split_to_real_and_imag_dfs(df, input_dim)

	df_real_normalized = _normalize_df_parallel(df_real)
	df_imag_normalized = _normalize_df_parallel(df_imag)

	return pd.concat([df_real_normalized, df_imag_normalized], axis=1)