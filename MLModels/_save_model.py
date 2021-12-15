from keras.models import load_model

def save_model(*args, **kwargs):
	
	directory = kwargs.get('directory')

	save_address = f"{directory}/" 
	model.save(save_address + "SavedModel.h5", save_format = 'h5')
	