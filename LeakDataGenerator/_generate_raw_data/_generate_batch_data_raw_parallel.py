from ._go_crazy import _go_crazy

from multiprocessing import Queue, Process
import multiprocessing as mp

def _generate_batch_data_raw_parallel(**params):
	batch_size_of_generator = params.get('batch_size_of_generator')

	results_queue = Queue()
	n_cores = mp.cpu_count()-2
	N_for_each_core = int(batch_size_of_generator/n_cores)

	pool = []
	for j in range (n_cores):
		worker = Process(target = _go_crazy, args = (results_queue,
													N_for_each_core, ))

		pool.append(worker)

	print('starting processes...')
	for worker in pool:
		worker.start()

	holder = []
	while any(worker.is_alive() for worker in pool):
		while not results_queue.empty():
			sample = results_queue.get()

			if not sample is None:
				holder.append(sample)

	print('finishing processes...')
	for worker in pool:
		worker.join()

	return holder[:batch_size_of_generator]


if __name__ == "__main__":

	_generate_batch_data_raw_parallel(N=500, batch_size_data = 100)
	print ("Data Generation done")