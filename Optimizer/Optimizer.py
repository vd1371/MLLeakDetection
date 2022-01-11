from ._load_data_for_optimization import _load_data_for_optimization

from .BaseOptimizer import BaseOptimizer
from ._optimize import optimize

class Optimizer(BaseOptimizer):

	def __init__(self, **params):
		super().__init__(**params)
		self.data = _load_data_for_optimization(**params)
		pop = params.get("population_size")
		self.probs = [(pop - i) / ((pop + 1)*pop/2) for i in range(pop)]

		self.location_precision = 1
		self.location_binary_length = 15
		self.size_precision = 3
		self.size_binary_length = 10

	def optimize(self):
		optimize(**self.__dict__)