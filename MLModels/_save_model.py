def _save_model(*args, **kwargs):
	
	directory = kwargs.get('directory')
	model_name = kwargs.get("model_name")

	save_address = f"{directory}/" 
	model.save(save_address + f"{model_name}-SavedModel.h5", save_format = 'h5')
	