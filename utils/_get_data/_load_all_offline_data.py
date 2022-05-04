import os
import pandas as pd

def _load_all_offline_data(**params):

	data_directory = params.get("data_directory")
	n_sections = params.get("n_sections")
	leak_pred = params.get("leak_pred")
	verbose = params.get("verbose")
	input_dim = params.get("input_dim")

	if verbose:
		print ("Trying to load files...")

	holder = []
	for file_name in os.listdir(data_directory)[:]:
		if leak_pred in file_name and '.csv' in file_name:
			data = pd.read_csv(f"{data_directory}/{file_name}", index_col = 0)
			holder.append(data)

			# if len(holder) > 0:
			# 	break

	data = pd.concat(holder, axis = 0)

	X = data.iloc[:, :input_dim]
	Y = data.iloc[:, input_dim:input_dim+n_sections]
	leak_info = data.iloc[:, input_dim+n_sections:]

	return X, Y, leak_info