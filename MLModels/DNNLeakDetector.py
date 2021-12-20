#Loading dependencies
import os
import requests
import io
import time

from ._log_hyperparameters import _log_hyperparameters
from ._construct_model import _construct_model
from .TrainLeakLocs import TrainLeakLocs
from ._BaseDNNLeakDetector import BaseDNNLeakDetector

from .TrainLeakLocs import TrainLeakLocs
from .TrainLeakSize import TrainLeakSize

class DNNLeakDetector(BaseDNNLeakDetector):

	def __init__(self, **params):
		super().__init__(**params)

	def _construct_model(self, *args, **kwargs):
		_log_hyperparameters(**self.__dict__)
		self.model = _construct_model(**self.__dict__)

	def run(self, *args, **kwargs):		
		if self.leak_pred == 'LeakLocs':
			TrainLeakLocs(**self.__dict__)
		else:
			TrainLeakSize(**self.__dict__)

if __name__ == "__main__":

	myDetector = DNNLeakDetector(epochs = 500)
	myDetector.run(n_rounds = 1000, warm_up = False, starting_batch = 0)
