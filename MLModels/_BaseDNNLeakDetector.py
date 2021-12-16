from utils import Logger

class BaseDNNLeakDetector:

	def __init__(self, **params):

		self.layers = params.get("layers")
		self.input_activation_func = params.get("input_activation_func")
		self.hidden_activation_func = params.get("hidden_activation_func")
		self.final_activation_func = params.get("final_activation_func")
		self.loss_func = params.get("loss_func")
		self.epochs = params.get("epochs")
		self.min_delta = params.get("min_delta")
		self.patience = params.get("patience")
		self.batch_size = params.get("batch_size")
		self.should_early_stop = params.get("should_early_stop")
		self.should_checkpoint = params.get("should_checkpoint")
	
		self.regul_type = params.get("regul_type")
		self.act_regul_type = params.get("act_regul_type")
		self.l = l2 if self.regul_type == 'l2' else l1
		self.actl = l1 if self.act_regul_type == 'l1' else l2
		self.reg_param = params.get("reg_param")
		self.dropout = params.get("dropout")
		self.optimizer = params.get("optimizer")
		self.random_state = params.get("random_state")

		self.split_size = params.get("split_size")

		self.data_directory = params.get("data_directory")
		self.directory = params.get("directory")
		self.log = Logger(address = f"{self.directory}/Log.log")
		self.input_dim = params.get("input_dim")
		self.output_dim = params.get("output_dim")

		self.n_rounds = params.get("n_rounds")
		self.warm_up = params.get("warm_up")
		self.starting_batch = params.get("starting_batch")