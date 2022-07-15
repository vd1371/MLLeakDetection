#Loading dependencies
from .BaseMLModel import BaseMLModel
from .DNN import _log_hyperparameters
from .DNN import _construct_model
from .DNN import train_leak_locs
from .DNN import train_leak_size

class DNNLeakDetector(BaseMLModel):

	def __init__(self, **params):
		super().__init__(**params)

	def _construct_model(self, **kwargs):
		_log_hyperparameters(**self.__dict__)
		self.model = _construct_model(**self.__dict__)

	def run(self, **kwargs):		
		if self.leak_pred == 'LeakLocs':
			train_leak_locs(**self.__dict__)
		elif self.leak_pred == 'LeakSize':
			train_leak_size(**self.__dict__)
