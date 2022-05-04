from ._go_crazy import _go_crazy

from multiprocessing import Queue, Process
import multiprocessing as mp

def _generate_batch_data_raw_parallel(**params):
	batch_size_of_generator = params.get('batch_size_of_generator')
	max_n_leaks = params.get('max_n_leaks')
	n_cores = params.get("n_cores")
	max_omeg_num = params.get("max_omeg_num")
	sigma_noise = params.get("sigma_noise")
	L = params.get('L')

	results_queue = Queue()
	N_for_each_core = int(batch_size_of_generator/n_cores)

	pool = []
	for j in range (n_cores):
		worker = Process(target = _go_crazy, args = (results_queue,
													N_for_each_core,
													max_n_leaks,
													L,
													max_omeg_num,
													sigma_noise,
													))

		pool.append(worker)

	print('starting processes...')
	for worker in pool:
		worker.start()

	holder = []
	while any(worker.is_alive() for worker in pool):
		while not results_queue.empty():
			sample = results_queue.get()
			if not sample is None:
				holder += sample

	print('finishing processes...')
	for worker in pool:
		worker.join()

	return holder[:batch_size_of_generator]


if __name__ == "__main__":

	_generate_batch_data_raw_parallel(N=500, batch_size_data = 100)
	print ("Data Generation done")