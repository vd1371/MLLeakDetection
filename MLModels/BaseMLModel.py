import os
from utils import Logger
import pprint
from datetime import datetime

class BaseMLModel:

	def __init__(self, **params):
		for k, v in params.items():
			setattr(self, k, v)

		now = datetime.now().strftime("%Y%m%d%H%M")
		self.report_directory = \
			os.path.join(".", 'reports', self.model, f"{self.leak_pred}-{now}")
		if not os.path.exists(self.report_directory):
			os.makedirs(self.report_directory)
		self.log = Logger(address = f"{self.report_directory}/Log.log")

		if self.model == "DNN":
			from tensorflow.keras.regularizers import l1, l2

			self.l = l2 if self.regul_type == 'l2' else l1
			self.actl = l1 if self.act_regul_type == 'l1' else l2

		self.log.info(pprint.pformat(self.__dict__))