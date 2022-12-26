from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

def _construct_model(**params):

	leak_pred_type = params.get("leak_pred")

	if leak_pred_type == "LeakLocs":

		model = LogisticRegression()

	elif leak_pred_type == "LeakSize":
		model = LinearRegression()
	else:
		raise ValueError ("leak_pred is not known. LeakLocs or LeakSize")

	return model