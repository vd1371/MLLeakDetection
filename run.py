import os

# from LeakDataGenerator import *
from MLModels import XGBLeakDetector
from MLModels.LinearLeakDetector import LinearLeakDetector
# from MLModels import CatBoostLeakDetector
# from Optimizer import Optimizer
from utils import leaks_distribution_hist


def run():

	pipe_and_leak_settings = {
		  'n_sections' : 25,
		  'max_n_leaks': 3,
		  'sigma_noise': 0.01,
		  'L': 2000,
		  'max_omeg_num': 20}

	general_settings = {
	  'data_directory' : './Data/',
	  'leak_pred': 'LeakSize',
	#   'leak_pred': 'LeakSize',
	  'starting_batch' : 0,
	  # "noises":[1,2,5,10,15,20],
	  'n_batches' : 1,
	  'n_sections' : 25,
	  'batch_size_of_generator': 1000,
	  'method': 'offline',
	  'verbose': True,
	  'n_cores': 6,
	  'input_dim' : pipe_and_leak_settings['max_omeg_num']*2,
	  'split_size': 0.2,
	  'random_seed' : 42,
	}

	imbalanced_figure_settings = {
	"data_directory" : './Data/',
	"num_hist_bins":2,
	"color":"Green",
	"dpi":300,
	"fig_height": 2,
	"fig_width": 3,
	"bar_width": 0.4,
	"font_size":10,
	"ticks_font_size":7,
	"marker_size":0,
	"line_width":1,
	"leak_pred":"LeakLocs",
	'input_dim': 40,
	'n_sections' : 25,
	'random_seed' : 42,

	}

	## On the Generator Side
	# Step 1: Run the generator
	# generate_data(**{**general_settings,
	# 					**pipe_and_leak_settings})

	# Step 2: Start the Server
	# run_server(server_class=HTTPServer,
	# 			handler_class=DataSender, addr="127.0.0.1", port=12000)

	## On the training side
	# Step 3: Run the below code
	# DNN_settings = {'layers' : [120,120, 120],
	# 		  'input_activation_func' : 'tanh',
	# 		  'hidden_activation_func' : 'relu',
	# 		  'epochs' : 200,
	# 		  'min_delta' : 0.00001,
	# 		  'patience' : 10,
	# 	      'batch_size' : 32,
	# 		  'should_early_stop' : False,
	# 		  'should_checkpoint' : False,
	# 	      'regul_type' : 'l2',
	# 		  'act_regul_type' : 'l1',
	# 		  'reg_param' : 0.01,
	# 		  'dropout' : 0.0,
	# 		  'optimizer' : 'adam',
	# 		  'random_state' : 42,
	# 		  'output_dim' : 1,
	# 		  'directory' : './Reports/DNN/',
	# 		  'model_name': "DNN",}

	# myDNNLeakDetector = DNNLeakDetector(**{**DNN_settings,
	# 										**general_settings})
	# myDNNLeakDetector._construct_model()
	# myDNNLeakDetector.run()

	# Step 4-1: Training XGB
	# XGB_settings = {'n_estimators' : 1000,
	# 				'learning_rate' : 0.3,
	# 				'max_depth' : 6,
	# 				'n_jobs' : -1,
	# 				'reg_alpha': 0.001,
	# 				'model_name': "XGB"}

	# myXGBLeakDetector = XGBLeakDetector(**{**XGB_settings,
	# 									 **general_settings})
	# myXGBLeakDetector._construct_model()
	# myXGBLeakDetector.run()

	linear_settings = {'model_name': "Linear"}
	myLinearModel = LinearLeakDetector(**{**linear_settings,
										 **general_settings})
	myLinearModel._construct_model()
	myLinearModel.run()

	# Step 4-2: Training catboost
	# cb_settings = {'iterations' : 100,
	# 				'learning_rate' : 0.1,
	# 				'depth' : 8,
	# 				'l2_leaf_reg' : 0.001,
	# 				# 'loss_function' : 'Logloss',
	# 				'loss_function' : 'RMSE',
	# 				'allow_writing_files' : False,
	# 				# 'eval_metric' : "Accuracy",
	# 				'eval_metric' : "RMSE",
	# 				'task_type' : 'CPU',
	# 				'verbose' : 400,
	# 				'boosting_type' : 'Ordered',
	# 				'thread_count' : -1,
	# 				'model_name': "CatBoost",}

	# myCatBoostLeakDetector = CatBoostLeakDetector(**{**cb_settings,
	# 	                                          **general_settings})
	# myCatBoostLeakDetector._construct_model()
	# myCatBoostLeakDetector.run()

	# Step 5: Optimze
	# GA_settings = {	'n_samples': 12,
	# 				'crossver_prob': 0.75,
	# 				'mutation_prob' : 0.05,
	# 				'population_size' : 200,
	# 				'n_generations' : 5000,
	# 				'n_elites' : 5,
	# 				'model_name': "GA",}

	# myOptimizer = Optimizer(**{**general_settings,
	# 						**pipe_and_leak_settings,
	# 						**GA_settings})
	# myOptimizer.optimize()

	# leaks_distribution_hist(6, **imbalanced_figure_settings)


if __name__ == "__main__":
	run()

	# os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

	print ("Done")
