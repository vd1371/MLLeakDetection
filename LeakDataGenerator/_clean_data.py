

def clean_data(df):

	# To get the length from the pipe chars
	# L = list(set(df.loc[:, 'L']))[0]

	df.drop(columns = ['L', "n_leaks", 'max_n_leaks', 'sound_speed',
						'diameter', 'Area_of_pipe', "friction_parameter"], inplace = True)

	idx = list(df.columns).index("X0")
	cols = list(df.columns)[idx:] + ['xL0', 'xL1', 'xL2', 'CdAl0', 'CdAl1', 'CdAl2']

	df = df[cols]

	return df