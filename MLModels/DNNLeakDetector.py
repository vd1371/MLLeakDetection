#Loading dependencies
from .DNN import _log_hyperparameters
from .DNN import _construct_model
from .DNN import BaseDNNLeakDetector
from .DNN import train_leak_locs
# from .TrainLeakSize import TrainLeakSize

class DNNLeakDetector(BaseDNNLeakDetector):

	def __init__(self, **params):
		super().__init__(**params)

	def _construct_model(self, *args, **kwargs):
		_log_hyperparameters(**self.__dict__)
		self.model = _construct_model(**self.__dict__)

	def run(self, *args, **kwargs):		
		if self.leak_pred == 'LeakLocs':
			train_leak_locs(**self.__dict__)
		else:
			train_leak_size(**self.__dict__)

if __name__ == "__main__":

	myDetector = DNNLeakDetector(epochs = 500)
	myDetector.run(n_rounds = 1000, warm_up = False, starting_batch = 0)
