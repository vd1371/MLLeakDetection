#Loading dependencies
from .BaseMLModel import BaseMLModel
from .RF import _log_hyperparameters
from .RF import _construct_model
from .RF import train_leak_locs
from .RF import train_leak_size

class RFLeakDetector(BaseMLModel):

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