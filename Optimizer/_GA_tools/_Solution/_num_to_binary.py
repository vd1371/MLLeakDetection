import numpy as np


def _num_to_binary(num, array_length, precision = 1):
	num = int(num * precision)
	result = [(num>>k)&1 for k in range(0, array_length)]
	result.reverse()
	return result


if __name__ == "__main__":

	binary = _num_to_binary(2000.87, 18, 10**precision)
	print (list(binary))
	print (_binary_to_num(binary))