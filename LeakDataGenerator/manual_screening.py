import matplotlib.pyplot as plt

from _generate_random_features import generate_random_features
from _h_d_measure import _h_d_measure
import time

def run():

	features = generate_random_features(random_location = False,
										random_size = False)
	
	plt.ion()

	for incre in range(2000):

		features['L'] = 1000 + incre
		print (features)

		params, wave = _h_d_measure(**features)

		y_real, y_imag = [], []

		for val in wave:
			y_real.append(val.real)
			y_imag.append(val.imag)

		plt.ion()
		plt.clf()
		x = [i for i in range(len(wave))]
		plt.plot(x, y_real, label = 'real')
		plt.plot(x, y_imag, label = 'imag')
		plt.grid(which='both')
		plt.legend()
		plt.draw()
		plt.pause(0.0001)
	

if __name__ == "__main__":
	run()