from utils import Logger

class BaseCatBoostLeakDetector:

	def __init__(self, **params):

		self.iterations = params.get("iterations")
		self.learning_rate = params.get("learning_rate")
		self.depth = params.get("depth")
		self.l2_leaf_reg = params.get("l2_leaf_reg")
		self.loss_function = params.get("loss_function")
		self.allow_writing_files = params.get("allow_writing_files")
		self.eval_metric = params.get("eval_metric")
		self.task_type = params.get("task_type")
		self.random_seed = params.get("random_seed")
		self.verbose = params.get("verbose")
		self.boosting_type = params.get("boosting_type")
		self.thread_count = params.get("thread_count")

		self.data_directory = params.get("data_directory")
		self.report_directory = params.get("report_directory")
		self.log = Logger(address = f"{self.report_directory}/Log.log")

		self.warm_up = params.get("warm_up")
		self.leak_pred = params.get("leak_pred")
		self.method = params.get("method")

		self.n_sections = params.get("n_sections")
		self.input_dim = params.get("input_dim")
		self.verbose = params.get("verbose")