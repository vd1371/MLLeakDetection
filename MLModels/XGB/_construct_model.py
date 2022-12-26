import xgboost as xgb

def _construct_model(**params):

	max_depth = params.get("max_depth")
	n_jobs = params.get("n_jobs")
	n_estimators = params.get("n_estimators")
	leak_pred_type = params.get("leak_pred")
	learning_rate = params.get("learning_rate")

	if leak_pred_type == "LeakLocs":

		model = xgb.XGBClassifier(n_estimators=n_estimators,
									max_depth=max_depth,
									learning_rate = 0.1,
									verbosity = 2,
	                        		n_jobs=n_jobs)

	elif leak_pred_type == "LeakSize":
		model = xgb.XGBRegressor(n_estimators=n_estimators,
									max_depth=max_depth,
									learning_rate = 0.1,
									verbosity = 2,
	                        		n_jobs=n_jobs)
	else:
		raise ValueError ("leak_pred is not known. LeakLocs or LeakSize")

	return model