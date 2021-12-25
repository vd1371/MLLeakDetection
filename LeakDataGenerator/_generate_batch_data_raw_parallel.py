from ._go_crazy import _go_crazy

from multiprocessing import Queue, Process
import multiprocessing as mp


def _generate_batch_data_raw_parallel(**params):
	N = params.get('N')
	batch_size_data = params.get('batch_size_data')

	results_queue = Queue()
	n_cores = mp.cpu_count()-2
	N_for_each_core = int(N/n_cores)

	pool = []
	for j in range (n_cores):
		worker = Process(target = _go_crazy, args = (results_queue, N_for_each_core, ))

		pool.append(worker)

	print('starting processes...')
	for worker in pool:
		worker.start()

	holder = []
	# done_workers = 0
	# batch_number = 0

	# start = time.time()
	while any(worker.is_alive() for worker in pool):
		while not results_queue.empty():
			sample = results_queue.get()

			if not sample is None:
				holder.append(sample)

	print('finishing processes...')
	for worker in pool:
		worker.join()

	# return holder[batch_size_data:]
	return holder

'''
def convert3(df, L = 2000):

	df.loc[:, 'xL0':'xL2'] = (df.loc[:, 'xL0':'xL2']/ L)


def convert_to_sections2(df, n_sections = 40, L = 2000):

	l_sections = int(L/n_sections)

	df.loc[:, 'xL0':'xL2'] = ((df.loc[:, 'xL0':'xL2']-0.0000001) / l_sections)

	# print (df)

	# Converting to integer
	cols = ['xL0', 'xL1', 'xL2']
	df[cols] = df[cols].applymap(np.int16)

	df.drop(columns = ['CdAl0', 'CdAl1', 'CdAl2'], inplace = True)

	return df, 4
'''


if __name__ == "__main__":

	# MC_sample(N=100)

	_generate_batch_data_raw_parallel(N=500, batch_size_data = 100)


	print ("Data Generation done")