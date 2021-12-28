import os
import pandas as pd

def _load_all_offline_data(**params):

	data_directory = params.get("data_directory")
	n_sections = params.get("n_sections")
	leak_pred = params.get("leak_pred")
	verbose = params.get("verbose")

	if verbose:
		print ("Trying to load files...")

	holder = []
	for file_name in os.listdir(data_directory):
		if leak_pred in file_name and '.csv' in file_name:
			data = pd.read_csv(f"{data_directory}/{file_name}", index_col = 0)
			holder.append(data)

	data = pd.concat(holder, axis = 0)
	data.reset_index(inplace = True)

	X, Y = data.iloc[:, :-n_sections], data.iloc[:, -n_sections:]

	return X, Y