import numpy as np 
from scipy import interp
from scipy.interpolate import LinearNDInterpolator
from numpy import maximum as npmax
from numpy import minimum as npmin
import matplotlib.pyplot as plt
import time



class Job_Search_SV(object):
	"""
	A class to store a given parameterization of the generalized
	job search model.

	The state process:
	w_t = eta_t + thet_t * xi_t, 
	log(thet_t) = rho* log(thet_{t-1}) + log(u_t),
	where:
		eta_t ~ LN(mu_eta, gam_eta) = v(.)
		xi_t ~ LN(0, gam_xi) = h(.)
		u_t ~ LN(0, gam_u) 

	The value function:
	v^*(w, thet) = max{u(w)/(1 - beta),
	                   c0+ beta*E[v^*(w',thet')|thet]}
    where:
    	E[v^*(w',thet')|thet] 
    	= int 
    	     v^*(w', thet')* f(thet'|thet)* v(eta')* h(xi')
    	  d(thet',eta',xi')

    	w' = eta' + thet' * xi'
    	f(thet'|thet) = LN(rho* thet, gam_u)

    The continuation value operator:
    Q psi(thet) 
      = c0 + 
        beta * E{max{u(w')/(1-beta),psi(thet')}|thet}
    where:
    	E{max{u(w')/(1-beta),psi(thet')}|thet}
    	= int 
    		 max{u(w')/(1-beta),psi(thet')}
    		 * f(thet'|thet)* v(eta')* h(xi')
    	  d(thet',eta',xi')

    Parameters
	----------
	beta : scalar(float), optional(default=0.95)
		   The discount factor
	c0_tilde : scalar(float), optional(default=1.)
		       The unemployment compensation
    sig : scalar(float), optional(default=2.5)
          The coefficient of relative risk aversion
    gam_u : scalar(float), optional(default=1e-4)
    		The variance of the shock process {u_t}
    gam_xi : scalar(float), optional(default=5e-4)
    		 The variance of the transient shock process 
    		 {xi_t}
    mu_eta : scalar(float), optional(default=0.)
             The mean of the process {eta_t}
    gam_eta : scalar(float), optional(default=1e-6.)
              The variance of the process {eta_t}
	thet_min : scalar(float), optional(default=1e-3)
	           The minimum of the grid for thet
	thet_max : scalar(float), optional(default=10.)
	           The maximum of the grid for thet
	thet_size : scalar(int), optional(default=200)
			    The number of grid points over thet
	mc_size : scalar(int), optional(default=10000)
		      The number of Monte Carlo samples
	"""

	def __init__(self, beta=.95, c0_tilde=.6, sig=2.5,
		         rho=.75, gam_u=1e-4, gam_xi=5e-4,
		         mu_eta=0., gam_eta=1e-6,
		         thet_min=1e-3, thet_max=10., thet_size=100,
		         w_min=1e-3, w_max=10., w_size=100,
		         mc_size=1000):
	    self.beta, self.c0_tilde, self.sig = beta, c0_tilde, sig 
	    self.rho, self.gam_u, self.gam_xi = rho, gam_u, gam_xi
	    self.mu_eta, self.gam_eta = mu_eta, gam_eta
	    self.thet_min, self.thet_max = thet_min, thet_max
	    self.thet_size = thet_size
	    self.w_min, self.w_max = w_min, w_max
	    self.w_size = w_size

	    # ============= make grids for CVI =============== #
	    self.grid_points = np.linspace(self.thet_min, 
	    	                           self.thet_max,
	    	                           self.thet_size)

	    # ============= make grids for VFI =============== #
	    self.thet_grids = np.linspace(self.thet_min, 
	    	                          self.thet_max,
	    	                          self.thet_size)
	    self.w_grids = np.linspace(self.w_min, self.w_max,
	    	                       self.w_size)
	    self.w_mesh, self.thet_mesh = np.meshgrid(self.w_grids,
	    	                                      self.thet_grids)
	    self.grid_points_vfi = \
	        np.column_stack((self.w_mesh.ravel(1), 
	    	                 self.thet_mesh.ravel(1)))
	    # MC draws
	    self.mc_size = mc_size
	    self.draws = np.random.randn(self.mc_size)


	def utility_func(self, x):
		"""
		The utility function (CRRA).
		"""
		sig = self.sig
		"""
		if sig == 1.:
			uw = np.log(x)
		else:
			uw = x**(1. - sig) / (1. - sig)
		"""
		uw = x**(1. - sig) / (1. - sig)
		#uw = np.log(x)
		return uw


	def Bellman_operator(self, v):
		"""
		The Bellman operator.
		"""
		beta, c0_tilde, rho = self.beta, self.c0_tilde, self.rho
		mu_eta, gam_eta = self.mu_eta, self.gam_eta
		gam_u, gam_xi = self.gam_u, self.gam_xi 
		draws = self.draws
		grid_points_vfi = self.grid_points_vfi
		utility_func = self.utility_func
		# interpolate to obtain a function
		v_interp = LinearNDInterpolator(grid_points_vfi, v)

		def v_f(x,y):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			"""
			x = npmax(npmin(x, self.w_max), self.w_min)
			y = npmax(npmin(y, self.thet_max), self.thet_min)
			return v_interp(x,y)

		N = len(v)
		new_v = np.empty(N)
		c0 = utility_func(c0_tilde) # c0

		for i in range(N):
			w, thet = grid_points_vfi[i, :]
			# MC samples: eta'
			eta_draws = np.exp(mu_eta+ np.sqrt(gam_eta)*draws)
			# MC samples: thet'
			thet_draws = np.exp(rho * np.log(thet) + \
				         np.sqrt(gam_u) * draws)
			# MC samples: xi'
			xi_draws = np.exp(np.sqrt(gam_xi) * draws)

			# MC samples: w'
			w_draws = eta_draws + thet_draws * xi_draws
			# MC samples: v(w',thet')
			integrand = v_f(w_draws, thet_draws)

			# the continuation value
			cvf = c0 + beta * np.mean(integrand)
			# the exit payoff
			exit_payoff = utility_func(w) / (1 - beta)

			new_v[i] = max(exit_payoff, cvf)

		return new_v
						

	def cvals_operator(self, psi):
		"""
		The continuation value operator.
		"""
		beta, c0_tilde, rho = self.beta, self.c0_tilde, self.rho
		mu_eta, gam_eta = self.mu_eta, self.gam_eta
		gam_u, gam_xi = self.gam_u, self.gam_xi 
		draws = self.draws
		grid_points = self.grid_points
		utility_func = self.utility_func
		# interpolate to obtain a function
		psi_interp = lambda x: interp(x, grid_points, psi)
        
		def psi_f(x):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			"""
			x = npmax(npmin(x, self.thet_max), self.thet_min)
			return psi_interp(x)

		N = len(psi)
		new_psi = np.empty(N)
		c0 = utility_func(c0_tilde)

		for i, thet in enumerate(grid_points):
			# MC samples: eta'
			eta_draws = np.exp(mu_eta+ np.sqrt(gam_eta)*draws)
			# MC samples: thet'
			thet_draws = np.exp(rho * np.log(thet) + \
				         np.sqrt(gam_u) * draws)
			# MC samples: xi'
			xi_draws = np.exp(np.sqrt(gam_xi) * draws)

			# MC samples: u(w')
			utils = utility_func(eta_draws+ thet_draws* xi_draws)

			# MC samples: r(w')
			integrand_1 = utils / (1. - beta)
			# MC samples: psi(thet')
			integrand_2 = psi_f(thet_draws)
			# MC samples: max{r(w'), psi(thet')}
			integrand = npmax(integrand_1, integrand_2)

			new_psi[i] = c0 + beta * np.mean(integrand)

		return new_psi


	def compute_fixed_point(self, Q, psi, error_tol=1e-4,
		                    max_iter=500, verbose=1):
	    """
	    Compute the fixed point.
	    """
	    error = error_tol + 1.
	    iteration = 0

	    while error > error_tol and iteration < max_iter:

	    	Qpsi = Q(psi)
	    	error = max(abs(Qpsi - psi))
	    	psi = Qpsi
	    	iteration += 1

	    	if verbose:
	    		print ("Computing iteration ", iteration, " with error", error)

	    return psi



"""
# ================================================================ #
#               CVI v.s VFI: different grid sizes                  #
# ================================================================ #

# This block provides a detailed comparison of VFI and CVI
# under different grid sizes of theta and w.
# Do not run this block with other ones at the same time.
# There are overlapping variable names. 

thet_list = [200, 200, 300, 300, 400, 400]
w_list = [200, 400, 200, 400, 200, 400]


# ================= Computation Time: CVI =================== #

print ("")
print ("CVI in progress ... ")

loops = 50
time_cvi = np.empty((loops, len(thet_list)))

for i in range(len(thet_list)):
	# start the event 
	jssv = Job_Search_SV(thet_size=thet_list[i], w_size=w_list[i],
	                     sig=1.)
	# initial guess of the solution
	psi_0 = np.ones(len(jssv.grid_points)) 
    
    # iterate via CVI and calculate the time taken of each loop
	for j in range(loops):
		start_cvi_ji = time.time() # start the clock
		psi_new = jssv.cvals_operator(psi_0)
		psi_0 = psi_new
		# time taken of (loop j, theta value i)
		time_cvi[j, i] = time.time() - start_cvi_ji 
	print ("Loop ", i+1, " finished ...", 
	       len(thet_list)-i-1, " remaining ...")

meantime_cvi = np.mean(time_cvi, 1)

# some key loops 
key_loops_cvi = [10, 20, 50]
# store the time taken at the selected key loops
time_cvi_keyloops = np.empty((len(key_loops_cvi), len(thet_list)))

# calculate the time taken at the selected key loops
for i in range(len(thet_list)):
	for j in range(len(key_loops_cvi)):
		time_cvi_keyloops[j,i] = np.sum(time_cvi[:key_loops_cvi[j], i])

# calculate the mean time taken at the selected key loops
meantime_cvi_keyloops = np.mean(time_cvi_keyloops, 1)


# ================= Computation Time: VFI =================== #

print ("")
print ("VFI in progress ... ")

loops_vfi = 50
time_vfi = np.empty((loops_vfi, len(thet_list)))

for i in range(len(thet_list)):
	# start the event 
	jssv = Job_Search_SV(thet_size=thet_list[i], w_size=w_list[i],
	                     sig=1.)
	# initial guess of the solution
	v_0 = np.ones(len(jssv.grid_points_vfi)) 
    
    # iterate via CVI and calculate the time taken of each loop
	for j in range(loops_vfi):
		start_vfi_ji = time.time() # start the clock
		v_new = jssv.Bellman_operator(v_0)
		v_0 = v_new
		# time taken of (loop j, theta value i)
		time_vfi[j, i] = time.time() - start_vfi_ji 
	print ("Loop ", i+1, " finished ...", 
	       len(thet_list)-i-1, " remaining ...")

meantime_vfi = np.mean(time_vfi, 1)

# some key loops 
key_loops_vfi = [10, 20, 50]
# store the time taken at the selected key loops
time_vfi_keyloops = np.empty((len(key_loops_vfi), len(thet_list)))

# calculate the time taken at the selected key loops
for i in range(len(thet_list)):
	for j in range(len(key_loops_vfi)):
		time_vfi_keyloops[j,i] = np.sum(time_vfi[:key_loops_vfi[j], i])

# calculate the mean time taken at the selected key loops
meantime_vfi_keyloops = np.mean(time_vfi_keyloops, 1)


# =============== Computation Time : VFI ================= #
print ("")
print ("----------------------------------------------")
print ("")
print ("Time taken under different grid sizes")
print ("")
print ("theta size : ", thet_list)
print ("")
print ("w size : ", w_list)
print ("")
print ("Key loops CVI : ", key_loops_cvi)
print ("")
print ("Key loops VFI : ", key_loops_vfi)
print ("")
print ("CVI : ", time_cvi_keyloops)
print ("")
print ("VFI : ", time_vfi_keyloops)
print ("")
print ("Average time CVI : ", meantime_cvi_keyloops)
print ("")
print ("Average time VFI : ", meantime_vfi_keyloops)
print ("")
print ("----------------------------------------------")

"""




# ================================================================ #
#             CVI v.s VFI: different parameter values              #
# ================================================================ #

# This block provides a detailed comparison of VFI and CVI
# under different sigma (risk aversion coefficient) and
# rho (autoregression coefficient of theta) values.
# Do not run this block with other ones at the same time.
# There are overlapping variable names. 

sig_list = [2., 2., 2., 3., 3., 3., 4., 4.]
rho_list = [.8, .7, .6, .8, .7, .6, .8, .7]

# ================== Computation Time: CVI =================== #

print ("")
print ("CVI in progress ... ")

loops_cvi = 50
time_cvi = np.empty((loops_cvi, len(sig_list)))

for i in range(len(sig_list)):
	# start the event
	jssv = Job_Search_SV(rho=rho_list[i], sig=sig_list[i],
		                 thet_size=300, w_size=300)	
	# compute the fixed point (continuation value)
	psi_0 = np.ones(len(jssv.grid_points)) # initial guess of the solution

	for j in range(loops_cvi):
		start_time_ji = time.time() # start the clock
		psi_new = jssv.cvals_operator(psi_0)
		psi_0 = psi_new
		time_cvi[j, i] = time.time() - start_time_ji
	print ("Loop ", i+1, " finished ...",
		   len(sig_list) - i - 1, "remaining ...")

meantime_cvi = np.mean(time_cvi, 1)

# some key loops 
key_loops_cvi = [10, 20, 50]
# store the time taken at the selected key loops
time_cvi_keyloops = np.empty((len(key_loops_cvi), len(sig_list)))

# calculate the time taken at the selected key loops
for i in range(len(sig_list)):
	for j in range(len(key_loops_cvi)):
		time_cvi_keyloops[j,i] = np.sum(time_cvi[:key_loops_cvi[j], i])

# calculate the mean time taken at the selected key loops
meantime_cvi_keyloops = np.mean(time_cvi_keyloops, 1)


# ================== Computation Time: VFI =================== #

print ("")
print ("VFI in progress ... ")

loops_vfi = 50
time_vfi = np.empty((loops_vfi, len(sig_list)))

for i in range(len(sig_list)):
	# start the event
	jssv = Job_Search_SV(rho=rho_list[i], sig=sig_list[i],
		                 thet_size=300, w_size=300)	
	# compute the fixed point (continuation value)
	v_0 = np.ones(len(jssv.grid_points_vfi)) # initial guess of the solution

	for j in range(loops_vfi):
		start_time_ji = time.time() # start the clock
		v_new = jssv.Bellman_operator(v_0)
		v_0 = v_new
		time_vfi[j, i] = time.time() - start_time_ji
	print ("Loop ", i+1, " finished ...",
		   len(sig_list) - i - 1, "remaining ...")

meantime_vfi = np.mean(time_vfi, 1)

# some key loops 
key_loops_vfi = [10, 20, 50]
# store the time taken at the selected key loops
time_vfi_keyloops = np.empty((len(key_loops_vfi), len(sig_list)))

# calculate the time taken at the selected key loops
for i in range(len(sig_list)):
	for j in range(len(key_loops_vfi)):
		time_vfi_keyloops[j,i] = np.sum(time_vfi[:key_loops_vfi[j], i])

# calculate the mean time taken at the selected key loops
meantime_vfi_keyloops = np.mean(time_vfi_keyloops, 1)



# =============== Computation Time : CVI v.s VFI ================= #
print ("")
print ("----------------------------------------------")
print ("")
print ("Time taken under different parameter values")
print ("")
print ("sigma values : ", sig_list)
print ("")
print ("rho values : ", rho_list)
print ("")
print ("Key loops CVI : ", key_loops_cvi)
print ("")
print ("Key loops VFI : ", key_loops_vfi)
print ("")
print ("CVI : ", time_cvi_keyloops)
print ("")
print ("VFI : ", time_vfi_keyloops)
print ("")
print ("Average time CVI : ", meantime_cvi_keyloops)
print ("")
print ("Average time VFI : ", meantime_vfi_keyloops)
print ("")
print ("----------------------------------------------")