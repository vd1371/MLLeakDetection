from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

def _construct_model(**params):

	iterations = params.get("iterations")
	learning_rate = params.get("learning_rate")
	depth = params.get("depth")
	l2_leaf_reg = params.get("l2_leaf_reg")
	loss_function = params.get("loss_function")
	allow_writing_files = params.get("allow_writing_files")
	eval_metric = params.get("eval_metric")
	task_type = params.get("task_type")
	random_seed = params.get("random_seed")
	verbose = params.get("verbose")
	boosting_type = params.get("boosting_type")
	thread_count = params.get("thread_count")
	
	leak_pred_type = params.get("leak_pred")

	if leak_pred_type == "LeakLocs":

		model = CatBoostClassifier(iterations = iterations,
                               learning_rate = learning_rate,
                               depth = depth,
                               l2_leaf_reg = l2_leaf_reg,
                               loss_function = loss_function,
                               allow_writing_files= allow_writing_files,
                               eval_metric = eval_metric,
                               task_type = task_type,
                               random_seed = random_seed,
                               verbose = verbose,
                               boosting_type = boosting_type,
                               thread_count = thread_count)

	elif leak_pred_type == "LeakSize":
		model = CatBoostRegressor(iterations = iterations,
                               learning_rate = learning_rate,
                               depth = depth,
                               l2_leaf_reg = l2_leaf_reg,
                               loss_function = loss_function,
                               allow_writing_files= allow_writing_files,
                               eval_metric = eval_metric,
                               task_type = task_type,
                               random_seed = random_seed,
                               verbose = verbose,
                               boosting_type = boosting_type,
                               thread_count = thread_count)

	else:
		raise ValueError ("leak_pred is not known. LeakLocs or LeakSize")

	return model