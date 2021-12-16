#Loading dependencies
import os
import requests
import io
import time

from ._get_data import get_data
from ._get_call_backs import _get_call_backs
from ._log_hyperparameters import _log_hyperparameters
from ._construct_model import _construct_model
from ._load_model import load_model
from ._save_model import save_model
from .run import run
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
		if self.modelling_type == "LeakLocs":
			TrainLeakLocs(self.model, **self.__dict__)
		else:
			raise ValueError("To MHK: Please fill this")

if __name__ == "__main__":

	# print (MC_sample_parallel(N=1000))

	myDetector = DNNLeakDetector(epochs = 500)
	myDetector.run(n_rounds = 1000, warm_up = False, starting_batch = 0)


