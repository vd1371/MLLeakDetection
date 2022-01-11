# Loading dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def h_d_measure():

	L = 2000        # pipe length. range 100:10'000
	no_L = 4     # number of leak. ok max 5
	# location of leaks: two leaks far away from each other
	xL = list(np.array([0.15, 0.25, 0.6, 0.8])*L)     

	a = 1200        # sound speed. 900:1200 important
	D = 0.5         # pipe diameter  not important
	A = np.pi*(D/2)**2     # area of pipe

	f = 0.02     # friction parameter

	CdAl = np.array([0.01, 0.03, 0.03, 0.01])*A    # size of leaks min 0.000001

	H0 = 20

	QL0 = CdAl*np.sqrt(2*9.81*H0)
	SS_L = QL0/(2*H0)
	
	omega_span = 2
	
	max_omeg_num = 20 #important

	omega_th = a*np.pi/(2*L)
	omega = omega_th * np.array(range(1, max_omeg_num+omega_span, omega_span))

	# steady-state discharge
	Q0 = 0.00153
	#R = 0


R = f*Q0/(9.8*D*A^2)

Reasonable_Zone=0.2*(2*pi*a/omega_th)  # the smaller the better, you may use 0.1 as the coefficient for more accuracy.

	mu = np.sqrt(-omega**2 + 1j*9.8*A*omega*R)/a
	Z = mu*a**2/(1j*omega*9.8*A)

	Lmat = np.array([[0, 1], [0, 0]])

	vx = [0] + xL + [L]

	hd_ff = []

	start = time.time()
	for ff in range(len(omega)):

		M_tr = np.array([[np.cosh(L*mu[ff]), -1/Z[ff]*np.sinh(L*mu[ff])],
						[-Z[ff]*np.sinh(L*mu[ff]), np.cosh(L*mu[ff])]])

		for ll in range(no_L):

			dx2 = vx[-1] - vx[ll+1]
			dx1 = vx[ll+1] - vx[0]

			mat1 =  np.array([[np.cosh(dx2*mu[ff]), -1/Z[ff]*np.sinh(dx2*mu[ff])], \
            				[-1*Z[ff]*np.sinh(dx2*mu[ff]), np.cosh(dx2*mu[ff])]])

			mat3 = np.array([[np.cosh(dx1*mu[ff]), -1/Z[ff]*np.sinh(dx1*mu[ff])], \
            				[-1*Z[ff]*np.sinh(dx1*mu[ff]), np.cosh(dx1*mu[ff])]])

			tmp = np.matmul(mat1, Lmat)
			tmp = np.matmul(tmp, mat3)
			M_tr = M_tr + tmp
		
		val = M_tr[1][0] / M_tr[0][0] 

		hd_ff.append(val)

	print (time.time()-start)


if __name__ == "__main__":

	h_d_measure()