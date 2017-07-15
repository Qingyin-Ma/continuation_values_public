import numpy as np
from numpy import maximum as npmax
from numpy import minimum as npmin
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import time



class Job_Search_adaptive(object):
	"""
	A class to store a given parameterization of the generalized
	job search model.

	The value function:
	v^*(w,mu,gam) = max{ u(w)/(1 - beta), 
	                     c0+ beta*E[v^*(w',mu',gam')|mu,gam]}
	
	The Bayesian updating process:
	Z_t = xi + eps_t, eps ~ N(0, gam_eps), iid
	w_t = e^{Z_t}
	prior: xi ~ N(mu, gam)
	posterior: xi|w' ~ N(mu', gam')
			   gam' = 1/(1/gam + 1/gam_eps)
			   mu' = gam'*(mu/gam + log(w')/gam_eps)

	Agents have constant relative risk aversion:
	u(w) = w^{1 - sig} / (1 - sig), 
	       if sig >=0 and sig is not equal to 1;
	u(w) = log(w), if sig = 1.

	Parameters
	----------
	beta : scalar(float), optional(default=0.95)
		   The discount factor
	c0_tilde : scalar(float), optional(default=0.05)
		       The unemployment compensation
    gam_eps : scalar(float), optional(default=1.)
              The variance of the shock process
    sig : scalar(float), optional(default=4.)
          the coefficient of relative risk aversion
    mu_min : scalar(float), optional(default=-10.)
             The minimum of the grid for mu
    mu_max : scalar(float), optional(default=10.)
             The maximum of the grid for mu
	mu_size : scalar(int), optional(default=50)
			 The number of grid points over mu
	gam_min : scalar(float), optional(default=1e-3)
	          The minimum of the grid for gam
	gam_max : scalar(float), optional(default=10.)
	          The maximum of the grid for gam
	gam_size : scalar(int), optional(default=50)
			  The number of grid points over gam
	w_min : scalar(float), optional(default=1e-3)
	        The minimum of the grid for w
	w_max : scalar(float), optional(default=10.)
	        The maximum of the grid for w
	w_size : scalar(int), optional(default=50)
	         The number of grid points over w
	mc_size : scalar(int), optional(default=10000)
		      The number of Monte Carlo samples
	"""
	def __init__(self, beta=0.95, c0_tilde=0.6,
		         gam_eps=1., sig=4.0, 
		         mu_min=-10., mu_max=10., mu_size=200,
		         gam_min= 1e-4, gam_max=10., gam_size=100,
		         w_min=1e-4, w_max=10., w_size=50,
		         mc_size=1000):
	    self.c0_tilde, self.beta = c0_tilde, beta
	    self.gam_eps, self.sig = gam_eps, sig
	    self.mu_min, self.mu_max = mu_min, mu_max
	    self.mu_size = mu_size
	    self.gam_min, self.gam_max = gam_min, gam_max
	    self.gam_size = gam_size
	    self.w_min, self.w_max = w_min, w_max
	    self.w_size = w_size
	    self.mc_size = mc_size
	    # make grids for mu
	    self.mu_grids = np.linspace(self.mu_min, self.mu_max, 
	    	                        self.mu_size)
	    # make grids for gam
	    self.gam_grids = np.linspace(self.gam_min, self.gam_max, 
	    	                         self.gam_size)
	    # make grids for w
	    self.w_grids = np.linspace(self.w_min, self.w_max,
	    	                       self.w_size)

	    # make grids for continuation value iteration (CVI)
	    self.mu_mesh, self.gam_mesh = np.meshgrid(self.mu_grids,
	    	                                      self.gam_grids)
	    self.grid_points = np.column_stack((self.mu_mesh.ravel(1),
	    	                                self.gam_mesh.ravel(1)))

	    # make grids for value function iteration (VFI)
	    self.w_mesh_vfi, self.mu_mesh_vfi, self.gam_mesh_vfi \
	        = np.meshgrid(self.w_grids, self.mu_grids, self.gam_grids)

	    self.grid_points_vfi=np.column_stack((self.w_mesh_vfi.ravel(1),
	    	                                  self.mu_mesh_vfi.ravel(1),
	    	                                  self.gam_mesh_vfi.ravel(1)))
	    # initial Monte Carlo draws
	    self.draws = np.random.randn(mc_size)


	def util_func(self,x):
		"""
		The (CRRA) utility function.
		"""
		sig = self.sig
		"""
		if sig == 1.:
			uw = np.log(x)
		else:
			uw = x**(1. - sig) / (1. - sig)
		"""
		#uw = np.log(x)
		uw = x**(1. - sig) / (1. - sig)
		return uw


	def r(self, x):
		"""
		The exit payoff (reward) function.
		r(w) = u(w) / (1 - beta).
		"""
		beta, sig = self.beta, self.sig
		"""
		if sig == 1.:
			rw = np.log(x) / (1. - beta)
		else:
			rw = x**(1. - sig) / ((1. - sig)*(1. - beta))
		"""
		#rw = np.log(x) / (1. - beta)
		rw = x**(1. - sig) / ((1. - sig)*(1. - beta))
		return rw


	def Bellman_operator(self, v):
		"""
		The Bellman operator.
		"""
		beta, c0_tilde = self.beta, self.c0_tilde
		gam_eps = self.gam_eps
		w_min, w_max = self.w_min, self.w_max
		mu_min, mu_max = self.mu_min, self.mu_max
		gam_min, gam_max = self.gam_min, self.gam_max
		mc_size, draws = self.mc_size, self.draws
		grid_points_vfi = self.grid_points_vfi
		v_interp = LinearNDInterpolator(grid_points_vfi, v)

		def v_f(x, y, z):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			"""
			x = npmin(npmax(x, w_min), w_max)
			y = npmin(npmax(y, mu_min), mu_max)
			z = npmin(npmax(z, gam_min), gam_max)
			return v_interp(x, y, z)

		N = len(v)
		new_v = np.empty(N)
		# the flow continuation payoff
		c0 = self.util_func(c0_tilde)

		for i in range(N):
			w, mu, gam = grid_points_vfi[i, :]
			# Bayesian updating of gam. "gam_prime" is a scalar.
			gam_prime = 1. / (1. / gam + 1. / gam_eps)
			# MC sampling : log(w') and w'
			# ln_prime, w_prime : an array of length mc_size
			lnw_prime = mu + np.sqrt(gam + gam_eps) * draws
			w_prime = np.exp(lnw_prime)
			# MC sampling : mu'; and
			# Bayesian updating of mu
			# mu_prime : an array of length mc_size
			mu_prime = gam_prime* (mu/gam + lnw_prime/gam_eps)

			# MC samples for v_f(w',mu',gam')
			integrand = v_f(w_prime, mu_prime, gam_prime)
			# the continuation value
			cvf = c0 + beta* np.mean(integrand)
			# the exit payoff 
			exit_payoff = self.r(w)

			new_v[i] = npmax(exit_payoff, cvf)

		return new_v



	def cval_operator(self, psi):
		"""
		The continuation value operator
		--------------------------------
		Qpsi = c0 + 
			   beta * int max{u(w')/(1-beta),phi(mu',gam')} 
			              * f(w'|mu,gam) dw'
		where:
			   u(w) is the CRRA utility
			   f(w'|mu, gam) = LN(mu, gam + gam_eps)
			   gam' = 1/(1/gam + 1/gam_eps)
			   mu' = gam' * (mu/gam + log(w')/gam_eps)

		The operator Q is a contraction mapping on 
		(b_l Z, rho_l) with unique fixed point psi^*,
		where:
		    l is the weight function defined by model
		    primitives.

		Parameters
		----------
		psi : array_like(float, ndim=1, length=len(grid_points))
			  An approximate fixed point represented as a one-dimensional
			  array.

		Returns
		-------
		new_psi : array_like(float, ndim=1, length=len(grid_points))
				  The updated fixed point.

		"""
		beta, gam_eps, c0_tilde = self.beta, self.gam_eps, self.c0_tilde
		mc_size, draws = self.mc_size, self.draws
		psi_interp = LinearNDInterpolator(self.grid_points, psi)
		
		def psi_f(x, y):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			"""
			x = npmin(npmax(x, self.mu_min), self.mu_max)
			y = npmin(npmax(y, self.gam_min), self.gam_max)
			return psi_interp(x, y)

		N = len(psi)
		new_psi = np.empty(N)
		# c0 = u(c0_tilde): write the unmemployment
		# compensation in the utility form
		c0 = self.util_func(self.c0_tilde) 

		for i in range(N):
			mu, gam = self.grid_points[i,:]
			# Bayesian updating of gam. "gam_prime" is a scalar.
			gam_prime = 1. / (1. / gam + 1. / gam_eps)
			# MC sampling : log(w')
			# ln_prime : an array of length mc_size
			lnw_prime = mu + np.sqrt(gam + gam_eps) * draws
			# MC sampling : mu'; and
			# Bayesian updating of mu
			# mu_prime : an array of length mc_size
			mu_prime = gam_prime* (mu/gam + lnw_prime/gam_eps)

			# MC sampling : r(w')
			integrand_1 = self.r(np.exp(lnw_prime))
			# MC sampling : psi_f(mu',gam')
			integrand_2 = psi_f(gam_prime * np.ones(mc_size),
				                mu_prime)
			integrand = npmax(integrand_1, integrand_2)	
			new_psi[i] = c0 + beta * np.mean(integrand) 

		return new_psi


	def compute_fixed_point(self, Q, psi, error_tol=1e-3,
		                    max_iter=500, verbose=1):
	    """
	    Compute the fixed point of the operator Q.
	    """
	    error = error_tol + 1.
	    iteration = 0

	    while error > error_tol and iteration < max_iter:
	    	Qpsi = Q(psi)
	    	error = max(abs(Qpsi - psi))
	    	psi = Qpsi
	    	iteration += 1

	    	if verbose:
	    		print ("Compute iteration ", iteration, " with error ", error)

	    return psi



# =================== Different grid sizes ======================= #

# =================== Computation Time : CVI ===================== #
print ("")
print ("CVI in progress ... ")

#mu_list_cvi = [20, 50, 50, 100, 200]
#gam_list_cvi = [20, 50, 50, 100, 200]

mu_list_cvi = [100]
gam_list_cvi = [100]

loops_cvi = 50
time_cvi = np.empty((loops_cvi, len(mu_list_cvi)))

for i in range(len(mu_list_cvi)):  
	# compute the continuation value via CVI
	jsa = Job_Search_adaptive(sig=3., mu_size=mu_list_cvi[i], 
		                      gam_size=gam_list_cvi[i])
	psi_0 = np.ones(len(jsa.grid_points)) # initial guess
    
    # iterate the Jovanovic operator "loops" times
	for j in range(loops_cvi):
		start_time_ji = time.time()
		psi_new = jsa.cval_operator(psi_0)
		psi_0 = psi_new
		time_cvi[j,i] = time.time() - start_time_ji
		print ("Loop ", j+1, " finished ...")

	#print ("Loop ", i+1, " finished... ", len(mu_list_cvi)-i-1, " remaining...")

meantime_cvi = np.mean(time_cvi, 1) # a vector of size (loops * 1)

# some key loops 
key_loops_cvi = [10, 20, 50]
# store the time taken at the selected key loops
time_cvi_keyloops = np.empty((len(key_loops_cvi), len(mu_list_cvi)))

# calculate the time taken at the selected key loops
for i in range(len(mu_list_cvi)):
	for j in range(len(key_loops_cvi)):
		time_cvi_keyloops[j,i] = np.sum(time_cvi[:key_loops_cvi[j], i])

# calculate the mean time taken at the selected key loops
meantime_cvi_keyloops = np.mean(time_cvi_keyloops, 1)


# =============== Computation Time : CVI ================= #
print ("")
print ("----------------------------------------------")
print ("")
print ("CVI : time taken under different grid sizes")
print ("")
print ("Key parameters : sig =", sig)
print ("")
print ("mu size : ", mu_list_cvi)
print ("")
print ("gam size : ", gam_list_cvi)
print ("")
print ("Key loops : ", key_loops_cvi)
print ("")
print ("CVI : ", time_cvi_keyloops)
print ("")
print ("Average time CVI : ", meantime_cvi_keyloops)
print ("")
print ("----------------------------------------------")




"""
# ============== Computation Time : VFI ================= #
print ("")
print ("VFI in progress ... ")

#w_list_vfi = [20, 20, 50, 50, 100]
#mu_list_vfi = [20, 50, 50, 100, 200]
#gam_list_vfi = [20, 50, 50, 100, 200]

w_list_vfi = [2, 2, 5, 5, 2]
mu_list_vfi = [2, 5, 5, 10, 2]
gam_list_vfi = [2, 5, 5, 10, 2]

loops_vfi = 3
time_vfi = np.empty((loops_vfi, len(mu_list_vfi)))

for i in range(len(mu_list_vfi)): 
	# compute the value function via VFI
	jsa = Job_Search_adaptive(sig=3., mu_size=mu_list_vfi[i], 
		        gam_size=gam_list_vfi[i], w_size=w_list_vfi[i])
	v_0 = np.ones(len(jsa.grid_points_vfi)) # initial guess
    
    # iterate the Bellman operator "loops_vfi" times
	for j in range(loops_vfi):
		start_time_ji = time.time()
		v_new = jsa.Bellman_operator(v_0)
		v_0 = v_new
		time_vfi[j,i] = time.time() - start_time_ji

	print ("Loop ", i+1, " finished... ", len(mu_list_vfi)-i-1, " remaining...")

meantime_vfi = np.mean(time_vfi, 1) # a vector of size (loops_vfi *1)

# some key loops 
key_loops_vfi = [1, 2, 3]
# store the time taken at the selected key loops
time_vfi_keyloops = np.empty((len(key_loops_vfi), len(mu_list_vfi)))

# calculate the time taken at the selected key loops
for i in range(len(mu_list_vfi)):
	for j in range(len(key_loops_vfi)):
		time_vfi_keyloops[j,i] = np.sum(time_vfi[:key_loops_vfi[j], i])

# calculate the mean time taken at the selected key loops
meantime_vfi_keyloops = np.mean(time_vfi_keyloops, 1)

# =============== Computation Time : VFI ================= #
print ("")
print ("----------------------------------------------")
print ("")
print ("VFI : time taken under different grid sizes")
print ("")
print ("w grid size : ", w_list_vfi)
print ("")
print ("mu grid size : ", mu_list_vfi)
print ("")
print ("gam grid size : ", gam_list_vfi)
print ("")
print ("Key loops : ", key_loops_vfi)
print ("")
print ("VFI : ", time_vfi_keyloops)
#print ("VFI : ", format(time_vfi_keyloops, '.5g'), "seconds")
print ("")
print ("Average time VFI: ", meantime_vfi_keyloops)
print ("")
print ("----------------------------------------------")
"""






# ================= Different parameter values =================== #

"""
# ============== Computation Time : CVI ================= #
print ("")
print ("CVI in progress ...")

sig_vals = [2., 3., 4., 5., 6.]
loops_cvi = 50
time_cvi = np.empty((loops_cvi, len(sig_vals)))

for i, sigm in enumerate(sig_vals):  
	# compute the continuation value via CVI
	jsa = Job_Search_adaptive(sig=sigm, mu_size=200, gam_size=200)
	psi_0 = np.ones(len(jsa.grid_points)) # initial guess
    
    # iterate the Jovanovic operator "loops" times
	for j in range(loops_cvi):
		start_time_ji = time.time()
		psi_new = jsa.cval_operator(psi_0)
		psi_0 = psi_new
		time_cvi[j,i] = time.time() - start_time_ji

	print ("Loop ", i+1, " finished... ", len(sig_vals)-i-1, " remaining...")

meantime_cvi = np.mean(time_cvi, 1) # a vector of size (loops * 1)

# some key loops 
key_loops_cvi = [10, 20, 50]
# store the time taken at the selected key loops
time_cvi_keyloops = np.empty((len(key_loops_cvi), len(sig_vals)))

# calculate the time taken at the selected key loops
for i in range(len(sig_vals)):
	for j in range(len(key_loops_cvi)):
		time_cvi_keyloops[j,i] = np.sum(time_cvi[:key_loops_cvi[j], i])

# calculate the mean time taken at the selected key loops
meantime_cvi_keyloops = np.mean(time_cvi_keyloops, 1)


# =============== Computation Time : CVI ================= #
print ("")
print ("----------------------------------------------")
print ("")
print ("CVI: different parameter values")
print ("")
print ("sig_vals : ", sig_vals)
print ("")
print ("Key loops CVI : ", key_loops_cvi)
print ("")
print ("Time taken CVI : ", time_cvi_keyloops)
print ("")
print ("Average time CVI: ", meantime_cvi_keyloops)
print ("")
print ("----------------------------------------------")
"""



"""
# ============== Computation Time : VFI ================= #
print ("")
print ("VFI in progress ... ")

sig_vals = [2., 3., 4., 5., 6.]
loops_vfi = 3
time_vfi = np.empty((loops_vfi, len(sig_vals)))

for i, sigm in enumerate(sig_vals):  
	# compute the value function via VFI
	jsa = Job_Search_adaptive(sig=sigm, mu_size=5, gam_size=5, w_size=5)
	v_0 = np.ones(len(jsa.grid_points_vfi)) # initial guess
    
    # iterate the Bellman operator "loops_vfi" times
	for j in range(loops_vfi):
		start_time_ji = time.time()
		v_new = jsa.Bellman_operator(v_0)
		v_0 = v_new
		time_vfi[j,i] = time.time() - start_time_ji

	print ("Loop ", i+1, " finished... ", len(sig_vals)-i-1, " remaining...")

meantime_vfi = np.mean(time_vfi, 1) # a vector of size (loops_vfi *1)

# some key loops 
key_loops_vfi = [1, 2, 3]
# store the time taken at the selected key loops
time_vfi_keyloops = np.empty((len(key_loops_vfi), len(sig_vals)))

# calculate the time taken at the selected key loops
for i in range(len(sig_vals)):
	for j in range(len(key_loops_vfi)):
		time_vfi_keyloops[j,i] = np.sum(time_vfi[:key_loops_vfi[j], i])

# calculate the mean time taken at the selected key loops
meantime_vfi_keyloops = np.mean(time_vfi_keyloops, 1)


# =============== Computation Time : VFI ================= #
print ("")
print ("----------------------------------------------")
print ("")
print ("")
print ("sig_vals : ", sig_vals)
print ("")
print ("VFI : ", time_vfi_keyloops)
print ("")
print ("Average time VFI: ", meantime_vfi_keyloops)
print ("")
print ("----------------------------------------------")
"""