import pandas as pd

from multiprocessing import Queue, Process
import multiprocessing as mp

from ._normalize_for_processes import _normalize_for_processes

def _normalize_df_parallel(df):

	results_queue = Queue()
	n_cores = mp.cpu_count()-2
	l_batches = int(len(df)/n_cores)

	pool = []
	for j in range (n_cores):
		tmp_df = df.iloc[j*l_batches: min((j+1)*l_batches, len(df)), :]

		worker = Process(target = _normalize_for_processes,
							args = (tmp_df,
									results_queue,))

		pool.append(worker)

	print('starting normalization processes...')
	for worker in pool:
		worker.start()

	holder = []
	while any(worker.is_alive() for worker in pool):
		while not results_queue.empty():
			sample = results_queue.get()

			if not sample is None:
				holder.append(sample)

	print('normalization done...')
	df = pd.concat(holder, axis = 0)

	return df