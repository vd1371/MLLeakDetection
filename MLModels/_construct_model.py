


def _construct_model(**params):

	riase ValueError("To MHK: Please complete this file")

	constructed = False
	if warm_up:
		try:
			load_model()
			constructed = True
			log.info("\n\n------------\nA trained model is loaded\n------------\n\n")
		except OSError:
			print ("The model is not trained before. No saved models found")

	if not constructed:
		# Creating the structure of the neural network
		model = _construct_network(**params)
		
		# A summary of the model
		stringlist = []
		model.summary(print_fn=lambda x: stringlist.append(x))
		short_model_summary = "\n".join(stringlist)
		log.info(short_model_summary)

	return model