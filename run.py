from LeakDataGenerator import *
from MLModels import DNNLeakDetector
from utils import *


def run():

	DNN_settings = {'layers' : [1000,1000],
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
			  'input_dim' : 50,
			  'output_dim' : 40}

	modelling_settings = {
			  'data_directory' : './Data/',
			  'directory' : './Reports/DNN/',
			  'n_rounds' : 1000,
			  'warm_up' : False,
			  'starting_batch' : 0,
			  'method': 'offline',
			  'verbose': True,
			  'leak_pred': 'LeakLocs'}

	myDNNLeakDetector = DNNLeakDetector(**{**DNN_settings,
											**modelling_settings})
	myDNNLeakDetector._construct_model()
	myDNNLeakDetector.run()


if __name__ == "__main__":
	run()

	print ("Done")