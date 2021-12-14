from LeakDataGenerator import *
from utils import *
from MLModels.DNNLeakDetector import get_data_prim, run


def run():

	# features = generate_random_features()
	# print (h_d_measure(features))

	# plot_varying_location()

	samples = generate_batch_data(N = 100000)
	df = convert_samples_to_df(samples)
	df = clean_data(df)
	df_leak_locs, df_leak_size = convert_to_sections(df)
	X_train, X_test, Y_train, Y_test = get_data_prim(df)










	convert_to_csv(df_leak_locs, df_leak_size, batch_number = 1000)





if __name__ == "__main__":
	run()

	print ("Done")