from ._generate_batch_data_raw import _generate_batch_data_raw
from ._convert_samples_to_df import convert_samples_to_df
from ._clean_data import clean_data
from ._convert_to_sections import convert_to_sections


def generate_batch_data_df(**params):

	samples = _generate_batch_data_raw(**params)
	df = convert_samples_to_df(samples)
	df = clean_data(df)
	df_leak_locs, df_leak_size = convert_to_sections(df)

	return df_leak_locs, df_leak_size

