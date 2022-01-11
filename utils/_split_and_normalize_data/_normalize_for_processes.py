import pandas as pd

def _normalize_for_processes(df, q_out):

	tmp_holder = []
	for idx, row in df.iterrows():
		tmp_holder.append((row.values-row.min())/(row.max()-row.min()))
	df = pd.DataFrame(tmp_holder, columns = df.columns, index = df.index)

	q_out.put(df)