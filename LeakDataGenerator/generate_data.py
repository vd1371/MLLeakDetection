import time

from ._generate_raw_data import generate_batch_data_raw

from ._convert_to_df_and_csv import convert_samples_to_df
from ._convert_to_df_and_csv import clean_data
from ._convert_to_df_and_csv import convert_to_sections
from ._convert_to_df_and_csv import convert_to_csv

def generate_data(**params):

	starting_batch = params.get("starting_batch")
	n_batches = params.get("n_batches")
	verbose = params.get("verbose")

	for batch_number in range(starting_batch, n_batches):

		samples = generate_batch_data_raw(**params)
		df = convert_samples_to_df(samples)
		df = clean_data(df)
		df_leak_locs, df_leak_size = convert_to_sections(df, **params)
		convert_to_csv (df_leak_locs, df_leak_size, batch_number)

		if verbose:
			print (f"Bacth {batch_number} is generated and saved.")
