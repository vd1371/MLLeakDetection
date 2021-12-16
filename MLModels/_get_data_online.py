

def get_data_online(batch_number):

	return 



	# 	downloaded = False
	# 	while not downloaded:
	# 		x = requests.get(f"http://158.132.126.58:8000/?batch_number={batch_number}").content.decode("utf-8")

	# 		if "NotFound" in x:
	# 			input("The data was not found. Make sure everything is alright")
	# 		else:
	# 			data = pd.read_csv(io.StringIO(x), header = 0, index_col = 0)
	# 			downloaded = True
	# 			print (batch_number, "is downloaded from server") 

	# 	X, Y, dates = data.iloc[:, :-3], data.iloc[:, -3:], data.index

	# 	X_train, X_test, Y_train, Y_test, \
	# 		dates_train, dates_test = train_test_split(X, Y, dates,
	# 													test_size = self.split_size,
	# 													shuffle = True,
	# 													random_state = self.random_state)

	# 	return X_train, X_test, Y_train, Y_test, dates_train, dates_test