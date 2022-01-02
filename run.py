from LeakDataGenerator import *
# from MLModels import DNNLeakDetector
from MLModels import RFLeakDetector
from MLModels import CatBoostLeakDetector

from utils import *

from WebUtils import *
from http.server import BaseHTTPRequestHandler, HTTPServer


def run():

	modelling_settings = {
			  'data_directory' : './Data/',
			  'report_directory': './reports/RF',
			  'leak_data_type' : 'Locs',
			  'starting_batch' : 0,
			  'n_batches' : 100,
			  'batch_size_of_generator': 30000,
			  'warm_up' : False,
			  'method': 'offline',
			  'verbose': True,
			  'leak_pred': 'LeakLocs',
			  'n_cores': 2,
			  'batch_size_data': 1000,
			  'N': 10000,
			  'n_sections' : 40,
			  'input_dim' : 50,}

	## On the Generator Side
	# Step 1: Run the generator
	# generate_data(**modelling_settings)

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
	# 		  'output_dim' : 40,
			  # 'directory' : './Reports/DNN/'}

	

	# myDNNLeakDetector = DNNLeakDetector(**{**DNN_settings,
	# 										**modelling_settings})
	# myDNNLeakDetector._construct_model()
	# myDNNLeakDetector.run()

	# Step 4-1: Training random forests
	rf_settings = {'n_estimators' : 500,
					'max_depth' : None,
					'min_samples_split' : 2,
					'min_samples_leaf' : 1,
					'max_features' : 'auto',
					'should_cross_val' : False,
					'n_jobs' : -1,}

	myRFLeakDetector = RFLeakDetector(**{**rf_settings,
										**modelling_settings})
	myRFLeakDetector._construct_model()
	myRFLeakDetector.run()

	# # Step 4-2: Training catboost
	# cb_settings = {'iterations' : 1000,
	# 				'learning_rate' : 0.0001,
	# 				'depth' : 9,
	# 				'l2_leaf_reg' : 0.000001,
	# 				'loss_function' : 'Logloss',
	# 				'allow_writing_files' : False,
	# 				'eval_metric' : "Accuracy",
	# 				'task_type' : 'CPU',
	# 				'random_seed' : 42,
	# 				'verbose' : 400,
	# 				'boosting_type' : 'Ordered',
	# 				'thread_count' : -1,}

	# myCatBoostLeakDetector = CatBoostLeakDetector(**{**cb_settings,
	# 	                                          **modelling_settings})
	# myCatBoostLeakDetector._construct_model()
	# myCatBoostLeakDetector.run()


if __name__ == "__main__":
	run()

	print ("Done")
