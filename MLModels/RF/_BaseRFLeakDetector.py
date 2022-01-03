from utils import Logger

from tensorflow.keras.regularizers import l1, l2

class BaseRFLeakDetector:

	def __init__(self, **params):

		self.n_estimators = params.get("n_estimators")
		self.max_depth = params.get("max_depth")
		self.min_samples_leaf = params.get("min_samples_leaf")
		self.min_samples_split = params.get("n_estimators")
		self.max_features = params.get("max_features")
		self.n_jobs = params.get("n_jobs")
		self.random_state = params.get("random_state")

		self.data_directory = params.get("data_directory")
		self.report_directory = params.get("report_directory")
		self.log = Logger(address = f"{self.report_directory}/Log.log")

		self.warm_up = params.get("warm_up")
		self.leak_pred = params.get("leak_pred")
		self.method = params.get("method")

		self.n_sections = params.get("n_sections")
		self.input_dim = params.get("input_dim")
		self.verbose = params.get("verbose")
		self.split_size = params.get("split_size")
