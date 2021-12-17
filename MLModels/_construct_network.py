from ._log_hyperparameters import _log_hyperparameters
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2


def _construct_network( **kwargs):
	# print(kwargs)
	# raise ValueError
	layers = kwargs.get('layers')
	input_dim = kwargs.get('input_dim')
	output_dim = kwargs.get('output_dim')
	input_activation_func = kwargs.get('input_activation_func')
	hidden_activation_func = kwargs.get('hidden_activation_func')
	final_activation_func = kwargs.get('final_activation_func')
	regul_type = kwargs.get('regul_type')
	act_regul_type = kwargs.get('act_regul_type')
	reg_param = kwargs.get('reg_param')
	dropout = kwargs.get('dropout')	
	loss_func = kwargs.get('loss_func')
	optimizer = kwargs.get('optimizer')
		
	l = l2 if regul_type == 'l2' else l1
	actl = l1 if act_regul_type == 'l1' else l2


	model = Sequential()
	
	print(kwargs.get('input_dim'))
	model.add(Dense(layers[0],
					input_shape = (input_dim,),
					activation = input_activation_func,
					kernel_regularizer = l(reg_param),
					activity_regularizer = actl(reg_param)))
	
	for ind in range(1,len(layers) + 1):
		model.add(Dense(layers[ind],
						activation = hidden_activation_func,
						kernel_regularizer = l(reg_param),
						activity_regularizer = actl(reg_param)))
		model.add(Dropout(dropout))
	
	model.add(Dense(output_dim, activation = final_activation_func))
	 
	# Compile model
	model.compile(loss=loss_func,
					optimizer=optimizer,
					metrics = ['accuracy'])

	return model