from keras.models import load_model

def load_model(*args, **kwargs):
	
	should_checkpoint = kwargs.get('should_checkpoint')
	directory = kwargs.get('directory')

	# load json and create model
	if should_checkpoint:
		model_type = 'BestModel'
	else:
		model_type = 'SavedModel'

	model = load_model(directory + "/" +  f"{model_type}.h5")

	return model
