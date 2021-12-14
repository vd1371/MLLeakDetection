import os
import pandas as pd

def get_data(*args, **kwargs):

	data_directory = kwargs.get("data_directory")
	batch_number = kwargs.get("batch_number")
	n_sections = kwargs.get("n_sections")
	leak_data_type = kwargs.get("leak_data_type")

	address = os.path.join(data_directory, f"Leak{leak_data_type}-{batch_number}.csv")
	data = pd.read_csv(address, index_col = 0)

	X, Y = data.iloc[:, :-n_sections], data.iloc[:, -n_sections:]

	return X, Y