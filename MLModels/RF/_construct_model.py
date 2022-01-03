from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

def _construct_model(**params):

	max_depth = params.get("max_depth")
	min_samples_split = params.get("min_samples_split")
	min_samples_leaf = params.get("min_samples_leaf")
	max_features = params.get("max_features")
	verbose = params.get("verbose")
	n_jobs = params.get("n_jobs")
	n_estimators = params.get("n_estimators")
	leak_pred_type = params.get("leak_pred")

	if leak_pred_type == "LeakLocs":

		model = RandomForestClassifier(n_estimators=n_estimators,
									max_depth=max_depth,
	                        		min_samples_split=min_samples_split,
	                        		min_samples_leaf=min_samples_leaf,
	                        		max_features=max_features,
	                        		bootstrap=True,
	                        		n_jobs=n_jobs, 
	                        		verbose=verbose)

	elif leak_pred_type == "LeakSize":
		model = RandomForestRegressor(n_estimators=n_estimators,
									max_depth=max_depth,
	                        		min_samples_split=min_samples_split,
	                        		min_samples_leaf=min_samples_leaf,
	                        		max_features=max_features,
	                        		bootstrap=True,
	                        		n_jobs=n_jobs, 
	                        		verbose=verbose)
	else:
		raise ValueError ("leak_pred is not known. LeakLocs or LeakSize")

	return model