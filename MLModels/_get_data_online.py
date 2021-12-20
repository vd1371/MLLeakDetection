

def get_data_online(**params):

	ip_address = params.get("ip_address")

	downloaded = False
	while not downloaded:
		resp = requests.get(f"http://{ip_address}/?batch_number={batch_number}").content.decode("utf-8")

		if "NotFound" in resp:
			input("The data was not found. Make sure everything is alright")
		else:
			data = pd.read_csv(io.StringIO(resp), header = 0, index_col = 0)
			downloaded = True
			print (batch_number, "is downloaded from server") 

	X, Y = data.iloc[:, :-n_sections], data.iloc[:, -n_sections:]

	return X, Y