import os
import pandas as pd

def get_data_offline(batch_number, **kwargs):

	data_directory = kwargs.get("data_directory")
	n_sections = kwargs.get("n_sections")
	leak_data_type = kwargs.get("leak_data_type")

	address = os.path.join(data_directory, f"Leak{leak_data_type}-{batch_number}.csv")
	data = pd.read_csv(address, index_col = 0)

	X, Y = data.iloc[:, :-n_sections], data.iloc[:, -n_sections:]

	return X, Y