import time

from ._generate_batch_data_df import _generate_batch_data_df
from ._convert_to_csv import convert_to_csv

def generate_data(**params):

	starting_batch = params.get("starting_batch")
	n_batches = params.get("n_batches")

	for batch_number in range(starting_batch, n_batches):

		df_leak_locs, df_leak_size = \
			_generate_batch_data_df(**params)
		convert_to_csv (df_leak_locs, df_leak_size, batch_number)
