# Loading dependencies
import numpy as np

def _head_measure_exact_model(**pipe_and_leak_params):

	a = 1200 # sound speed. 900:1200 important
	D = 0.5 # pipe diameter  not important 
	A = np.pi*(D/2)**2 # area of pipe
	f = 0.02 # friction parameter

	L = pipe_and_leak_params.get("L")
	xL = pipe_and_leak_params.get('xL')
	CdAl = pipe_and_leak_params.get('CdAl') *0.5*A
	max_omeg_num = pipe_and_leak_params.get("max_omeg_num")

	H0 = 20
	QL0 = CdAl*np.sqrt(2*9.81*H0)
	SS_L = QL0/(2*H0)
	omega_span = 2
	omega_th = a*np.pi/(2*L)
	omega = omega_th * np.array(range(1, max_omeg_num*2, omega_span))

	Q0 = 0.00153
	R = f*Q0/(9.8*D*A**2)
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

	return np.array(hd_ff)