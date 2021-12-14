#Loading deoendencies
import os
import numpy as np
import pandas as pd
import time 

from LeakDataGenerator import _h_d_measure


from multiprocessing import Queue, Process
import multiprocessing as mp

def go_crazy(N, q_out):

	for i in range(N):
		q_out.put(h_d_measure())

def MC_sample_parallel(N = 100000, batch_size = 1000):

	results_queue = Queue()
	n_cores = mp.cpu_count()-2
	N_for_each_core = int(N/n_cores)

	pool = []
	for j in range (n_cores):
		worker = Process(target = go_crazy, args = (N_for_each_core, results_queue, ))
		pool.append(worker)

	print('starting processes...')
	for worker in pool:
		worker.start()

	all_samples = []
	done_workers = 0
	batch_number = 0

	start = time.time()
	while any(worker.is_alive() for worker in pool):

		while not results_queue.empty():
			sample = results_queue.get()

			if not sample is None:
				all_samples.append(sample)

		# Saving each batch
		if len(all_samples) > batch_size:
			batch_number += 1

			convert_to_csv(all_samples[:batch_size], batch_number)
			
			print (f'Batch number {batch_number} is done in {time.time()-start:.2f}')
			start = time.time()

			all_samples = all_samples[batch_size:]

	print('finishing processes...')
	for worker in pool:
		worker.join()

def convert3(df, L = 2000):

	df.loc[:, 'xL0':'xL2'] = (df.loc[:, 'xL0':'xL2']/ L)


def convert_to_sections2(df, sects = 200, L = 2000):

	l_sects = int(L/sects)

	df.loc[:, 'xL0':'xL2'] = ((df.loc[:, 'xL0':'xL2']-0.0000001) / l_sects)

	print (df)

	# Converting to integer
	cols = ['xL0', 'xL1', 'xL2']
	df[cols] = df[cols].applymap(np.int16)

	df.drop(columns = ['CdAl0', 'CdAl1', 'CdAl2'], inplace = True)

	return df, 4



if __name__ == "__main__":

	# MC_sample(N=100)

	MC_sample_parallel(N=500, batch_size = 100)


	print ("Data Generation done")