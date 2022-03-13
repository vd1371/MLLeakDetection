from ._optimize_linear import _optimize_linear

import pandas as pd
from multiprocessing import Queue, Process
import multiprocessing as mp

def _optimize_parallel(data, **params):

	results_queue = Queue()
	n_cores = mp.cpu_count()-2
	l_batches = int(len(data[0])/n_cores)

	pool = []
	for j in range (n_cores):
		tmp_data = (data[0].iloc[j*l_batches: min((j+1)*l_batches, len(data[0])),:],
					data[1].iloc[j*l_batches: min((j+1)*l_batches, len(data[1])),:],
					data[2].iloc[j*l_batches: min((j+1)*l_batches, len(data[2])),:])

		worker = Process(target = _optimize_linear,
							args = (tmp_data,
									params,
									results_queue,
									))

		pool.append(worker)

	print('starting optimization processes...')
	for worker in pool:
		worker.start()

	holder = []
	while any(worker.is_alive() for worker in pool):
		while not results_queue.empty():
			sample = results_queue.get()

			if not sample is None:
				holder.append(sample)

	return holder[0]