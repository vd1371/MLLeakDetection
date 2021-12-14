from ._generate_batch_data_raw import generate_batch_data_raw
from ._convert_samples_to_df import convert_samples_to_df
from ._clean_data import clean_data
from ._convert_to_sections import convert_to_sections


def _generate_batch_data_df(N):

	samples = generate_batch_data_raw(N = 100000)
	df = convert_samples_to_df(samples)
	df = clean_data(df)
	df_leak_locs, df_leak_size = convert_to_sections(df)

	return df_leak_locs, df_leak_size

