from LeakDataGenerator import *
from MLModels import DNNLeakDetector
from utils import *

from WebUtils import *
from http.server import BaseHTTPRequestHandler, HTTPServer


def run():

	modelling_settings = {
			  'data_directory' : './Data/',
			  'directory' : './Reports/DNN/',
			  'leak_data_type' : 'Locs',
			  'starting_batch' : 0,
			  'n_batches' : 3000,
			  'batch_size_of_generator': 10000,
			  'warm_up' : False,
			  'method': 'offline',
			  'verbose': True,
			  'leak_pred': 'LeakLocs',
			  'n_cores': 2,
			  'batch_size_data': 1000,
			  'N': 10000}

	## On the Generator Side
	# Step 1: Run the generator
	generate_batch_data_df(**modelling_settings) #df shape = (9996,90) instead of (10000,90). maybe we should fix this

	# Step 2: Start the Server
	# run_server(server_class=HTTPServer,
	# 			handler_class=DataSender, addr="127.0.0.1", port=12000)

	## On the training side
	# Step 3: Run the below code
	# DNN_settings = {'layers' : [1000,1000],
	# 		  'input_activation_func' : 'tanh',
	# 		  'hidden_activation_func' : 'relu',
	# 		  'final_activation_func' : 'softmax',
	# 		  'loss_func' : 'categorical_crossentropy',
	# 		  'epochs' : 500,
	# 		  'min_delta' : 0.00001,
	# 		  'patience' : 10,
	# 	      'batch_size' : 32,
	# 		  'should_early_stop' : False,
	# 		  'should_checkpoint' : False,
	# 	      'regul_type' : 'l2',
	# 		  'act_regul_type' : 'l1',
	# 		  'reg_param' : 0.01,
	# 		  'dropout' : 0.2,
	# 		  'optimizer' : 'adam',
	# 		  'random_state' : 165,
	# 		  'split_size' : 0.2,
	# 		  'input_dim' : 50,
	# 		  'output_dim' : 40,
	# 		  'n_sections' : 40}

	

	# myDNNLeakDetector = DNNLeakDetector(**{**DNN_settings,
	# 										**modelling_settings})
	# myDNNLeakDetector._construct_model()
	# myDNNLeakDetector.run()


if __name__ == "__main__":
	run()

	print ("Done")
