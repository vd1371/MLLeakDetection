import numpy as np

def _binary_to_num(arr, precision = 3):
	'''Converts binary arrays to numbers
	
	The underlying assumption in validation is that the accuracy
	of results will be by 2 decimal points
	The arr shape is like: (n_assets, n_elements, mrr_for_element)
	'''
	out_str = ""
	for val in arr:
		out_str += str(val)
	return int(out_str, 2)/10**precision

if __name__ == "__main__":

	print (_binary_to_num([1 for i in range(10)]))