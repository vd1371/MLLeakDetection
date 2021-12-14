from .log_hyperparameters import log_hyperparameters
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential


def _construct_model(*args, **kwargs):

	log_hyperparameters()

	model = Sequential()
	
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