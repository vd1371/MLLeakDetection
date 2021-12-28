import logging as log
import pprint

def _log_hyperparameters(**kwargs):
	
	layers = kwargs.get('layers')
	input_activation_func = kwargs.get('input_activation_func')
	hidden_activation_func = kwargs.get('hidden_activation_func')
	final_activation_func = kwargs.get('final_activation_func')
	loss_func = kwargs.get('loss_func')
	epochs = kwargs.get('epochs')
	min_delta = kwargs.get('min_delta')
	patience = kwargs.get('patience')
	batch_size = kwargs.get('batch_size')
	should_early_stop = kwargs.get('should_early_stop')
	should_checkpoint = kwargs.get('should_checkpoint')
	regul_type = kwargs.get('regul_type')
	act_regul_type = kwargs.get('act_regul_type')
	reg_param = kwargs.get('reg_param')
	dropout = kwargs.get('dropout')
	optimizer = kwargs.get('optimizer')
	random_state = kwargs.get('random_state')
	split_size = kwargs.get('split_size')

	log.info(pprint.pformat({'layers': layers,
							'input_activation_func': input_activation_func,
							'hidden_activation_func': hidden_activation_func,
							'final_activation_func': final_activation_func,
							'loss_func': loss_func,
							'epochs': epochs,
							'min_delta': min_delta,
							'patience': patience,
							'batch_size': batch_size,
							'should_early_stop': should_early_stop,
							'should_checkpoint': should_checkpoint,
							'regul_type': regul_type,
							'act_regul_type': act_regul_type,
							'reg_param': reg_param,
							'dropout': dropout,
							'optimizer': optimizer,
							'random_state': random_state,
							'split_size': split_size,
							}))