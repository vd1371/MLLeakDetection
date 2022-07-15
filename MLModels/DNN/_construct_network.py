from ._log_hyperparameters import _log_hyperparameters
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.constraints import NonNeg 

def _construct_network( **params):
	layers = params.get('layers')
	input_dim = params.get('input_dim')
	output_dim = params.get('output_dim')
	input_activation_func = params.get('input_activation_func')
	hidden_activation_func = params.get('hidden_activation_func')
	regul_type = params.get('regul_type')
	act_regul_type = params.get('act_regul_type')
	reg_param = params.get('reg_param')
	dropout = params.get('dropout')	
	optimizer = params.get('optimizer')
	leak_pred = params.get("leak_pred")
		
	l = l2 if regul_type == 'l2' else l1
	actl = l1 if act_regul_type == 'l1' else l2


	model = Sequential()
	
	model.add(Dense(layers[0],
					input_shape = (input_dim,),
					activation = input_activation_func,
					kernel_regularizer = l(reg_param),
					activity_regularizer = actl(reg_param)))
	
	for ind in range(1,len(layers)):
		model.add(Dense(layers[ind],
						activation = hidden_activation_func,
						kernel_regularizer = l(reg_param),
						activity_regularizer = actl(reg_param)))
		model.add(Dropout(dropout))
	
	if leak_pred == "LeakLocs":
		model.add(Dense(output_dim, activation = 'softmax'))

	elif leak_pred == "LeakSize":
		model.add(Dense(output_dim, activation="sigmoid"))

	# Compile models
	if leak_pred == "LeakLocs":
		model.compile(loss='binary_crossentropy',
						optimizer=optimizer,
						metrics = ['accuracy'])

	elif leak_pred == "LeakSize":
		model.compile(loss='mean_squared_error',
						optimizer=optimizer)

	return model