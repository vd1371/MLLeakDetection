# Loading dependencies
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import time

from copy import deepcopy

def h_d_measure(params):

	L = params.get('L')
	xL = params.get('xL')
	a = params.get('sound_speed')
	A = params.get('Area_of_pipe')
	CdAl = params.get('CdAl') *0.5*A

	H0 = 20

	QL0 = CdAl*np.sqrt(2*9.81*H0)
	SS_L = QL0/(2*H0)
	
	omega_span = 2
	max_omeg_num = 25

	omega_th = a*np.pi/(2*L)
	omega = omega_th * np.array(range(1, max_omeg_num*2, omega_span))

	# steady-state discharge
	Q0 = 0.00153
	R = 0
	mu = np.sqrt(-omega**2 + 1j*9.8*A*omega*R)/a
	Z = mu*a**2/(1j*omega*9.8*A)

	Lmat = np.array([[0, 1], [0, 0]])

	vx = [0] + xL + [L]
	hd_ff = []

	for ff in range(len(omega)):

		M_tr = np.array([[np.cosh(L*mu[ff]), -1/Z[ff]*np.sinh(L*mu[ff])],
						[-Z[ff]*np.sinh(L*mu[ff]), np.cosh(L*mu[ff])]])

		for ll in range(len(CdAl)):

			dx2 = vx[-1] - vx[ll+1]
			dx1 = vx[ll+1] - vx[0]

			mat1 =  np.array([[np.cosh(dx2*mu[ff]), -1/Z[ff]*np.sinh(dx2*mu[ff])], \
							[-1*Z[ff]*np.sinh(dx2*mu[ff]), np.cosh(dx2*mu[ff])]])

			mat3 = np.array([[np.cosh(dx1*mu[ff]), -1/Z[ff]*np.sinh(dx1*mu[ff])], \
							[-1*Z[ff]*np.sinh(dx1*mu[ff]), np.cosh(dx1*mu[ff])]])

			tmp = np.matmul(mat1, Lmat)
			tmp = np.matmul(tmp, mat3)
			M_tr = M_tr + (-SS_L[ll]) * tmp
		
		val = M_tr[1][0] / M_tr[0][0] 

		hd_ff.append(val)

	return (params, hd_ff)

if __name__ == "__main__":

	for _ in range(10000):
		plotter()