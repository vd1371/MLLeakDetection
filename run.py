from LeakDataGenerator import *
from utils import *


def run():

	params = {'layers' : [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
			  'input_activation_func' : 'tanh',
			  'hidden_activation_func' : 'relu',
			  'final_activation_func' : 'softmax',
			  'loss_func' : 'categorical_crossentropy',
			  'epochs' : 500,
			  'min_delta' : 0.00001,
			  'patience' : 10,
		      'batch_size' : 32,
			  'should_early_stop' : False,
			  'should_checkpoint' : False,
		      'regul_type' : 'l2',
			  'act_regul_type' : 'l1',
			  'reg_param' : 0.01,
			  'dropout' : 0.2,
			  'optimizer' : 'adam',
			  'random_state' : 165,
			  'split_size' : 0.2,
			  'data_directory' : './Data/',
			  'directory' : './Reports/DNN/',
			  'input_dim' : 50,
			  'output_dim' : 40,
			  'n_rounds' : 1000,
			  'warm_up' : False,
			  'starting_batch' : 0}

	myDNNLeakDetector = DNNLeakDetector(**params)
	myDNNLeakDetector.run()

	# features = generate_random_features()
	# print (h_d_measure(features))

	# plot_varying_location()

	# samples = generate_batch_data(N = 100000)
	# df = convert_samples_to_df(samples)
	# df = clean_data(df)
	# df_leak_locs, df_leak_size = convert_to_sections(df)

	# convert_to_csv(df_leak_locs, df_leak_size, batch_number = 1000)





if __name__ == "__main__":
	run()

	print ("Done")