# Loading dependencies
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import time

from copy import deepcopy

from LeakDataGenerator import _h_d_measure

def plot_varying_location():

	first = True
	for i in range(100000):

		var = 200 + 0.1 * i
		params =  {"L": 2000,
			'no_L': 3,
			'max_L': 3,
			'xL': [100, var, 1814.97],
			'sound_speed': 1200,
			"diameter": 0.5,
			'Area_of_pipe': 0.196349,
			'friction_parameter': 0,
			'CdAl': np.array([0.1, 0.1, 0.1])
		}

		params, hd_ff = h_d_measure(params = params)

		if first:
			hd_ff_first = deepcopy(hd_ff)
			first = False

		print (var, f"{np.real(np.corrcoef(hd_ff, hd_ff_first)[0][1]):.2f}")

		plt.ion()
		plt.clf()
		plt.title(f'At {var}')
		plt.xlabel('ff')
		

		xticks = [i for i in range(25)]
		plt.plot(xticks, np.imag(hd_ff), label = 'imag')
		plt.plot(xticks, np.real(hd_ff), label = 'real')
		
		plt.legend()
		plt.grid(True, which = 'both')
		plt.draw()
		plt.pause(0.0001)


if __name__ == "__main__":

	for _ in range(10000):
		plotter()