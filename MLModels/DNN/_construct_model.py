from ._load_model import _load_model
from ._construct_network import _construct_network

from utils import Logger

def _construct_model(**kwargs):

	# raise ValueError("To MHK: Please complete this file")
	warm_up = kwargs.get('warm_up')
	log = kwargs.get("log")

	constructed = False
	if warm_up:
		try:
			_load_model()
			constructed = True
			log.info("\n\n------------\nA trained model is loaded\n------------\n\n")
		except OSError:
			print ("The model is not trained before. No saved models found")

	if not constructed:
		# Creating the structure of the neural network
		model = _construct_network(**kwargs)
		
		# A summary of the model
		stringlist = []
		model.summary(print_fn=lambda x: stringlist.append(x))
		short_model_summary = "\n".join(stringlist)
		log.info(short_model_summary)

	return model