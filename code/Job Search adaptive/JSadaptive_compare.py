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
		if sig == 1.:
			uw = np.log(x)
		else:
			uw = x**(1. - sig) / (1. - sig)
		return uw


	def r(self, x):
		"""
		The exit payoff (reward) function.
		r(w) = u(w) / (1 - beta).
		"""
		beta, sig = self.beta, self.sig
		if sig == 1.:
			rw = np.log(x) / (1. - beta)
		else:
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
			# the flow continuation payoff
			c0 = self.util_func(c0_tilde)
			# the continuation value
			cvf = c0 + beta* np.mean(integrand)
			# the exit payoff 
			exit_payoff = self.r(w)

			new_v[i] = npmax(exit_payoff, cvf)

		return new_v


	def compute_cvf(self, v):
		"""
		Compute the continuation value based on 
		the fixed point of the Bellman operator.
		"""
		beta, c0_tilde = self.beta, self.c0_tilde
		gam_eps = self.gam_eps
		w_min, w_max = self.w_min, self.w_max
		mu_min, mu_max = self.mu_min, self.mu_max
		gam_min, gam_max = self.gam_min, self.gam_max
		grid_points = self.grid_points
		grid_points_vfi = self.grid_points_vfi
		mc_size, draws = self.mc_size, self.draws

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

		N = len(grid_points)
		cvf = np.empty(N)

		for i in range(N):
			mu, gam = grid_points[i, :]
			# Bayesian updating of gam. "gam_prime" is a scalar.
			gam_prime = 1. / (1. / gam + 1. / gam_eps)
			# MC sampling : w'
			lnw_prime = mu + np.sqrt(gam + gam_eps) * draws
			w_prime = np.exp(lnw_prime)
			# MC sampling : mu'
			mu_prime = gam_prime * (mu/gam + lnw_prime/gam_eps)

			# MC sampling : v(w', mu', gam')
			integrand = v_f(w_prime, mu_prime, gam_prime)
			# the flow continuation payoff
			c0 = self.util_func(c0_tilde)
			# the continuation value
			cvf[i] = c0 + beta * np.mean(integrand)

		return cvf


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
			# c0 = u(c0_tilde): write the unmemployment
			# compensation in the utility form
			c0 = self.util_func(self.c0_tilde) 
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


	def res_rule_func(self, y):
		"""
		Recover the reservation rule based on the 
		continuation value function.
		"""
		sig, beta = self.sig, self.beta

		if sig == 1.:
			res_rule = np.exp(y* (1. - beta))
		else:
			base = (y * (1. - beta)* (1. - sig))
			res_rule = base**(1. / (1. - sig))

		return res_rule


	def res_utility(self, y):
		"""
		Recover the reservation utility based on the
		continuation value function.
		"""
		return y * (1 - self.beta)




# ============== Computation Time : CVI ================= #
print ("")
print ("CVI in progress")

sig_vals = [3., 4., 5., 6.]
time_taken = np.empty(4) 

for i, sigm in enumerate(sig_vals):
	start_cvi = time.time() # starting time of iteration i

	# computing the continuation value via CVI
	jsa = Job_Search_adaptive(sig=sigm)
	psi_0 = np.ones(len(jsa.grid_points))
	psi_star = jsa.compute_fixed_point(jsa.cval_operator, psi_0, \
		                               verbose=0)

	# calculate the time taken for iteration i
	time_taken[i] = time.time() - start_cvi

	print ("Loop ", i+1, " finished... ", 4-i-1, " remaining...")

time_cvi = np.mean(time_taken)


# =============== Computation Time : CVI ================= #
print ("")
print ("----------------------------------------------")
print ("")
print ("Average computation time: ")
print ("")
print ("CVI : ", format(time_cvi, '.5g'), "seconds")
print ("")
print ("----------------------------------------------")





"""
# This block is used to compare the computation time 
# of CVI and VFI. Do not run this block with other ones 
# in one simulation, as there are overlapping variable names.

# ============== Computation Time : CVI ================= #
print ("")
print ("CVI in progress")

start_cvi = time.time()

jsa = Job_Search_adaptive()

# compute the continuation value via CVI
psi_0 = np.ones(len(jsa.grid_points))
psi_star = jsa.compute_fixed_point(jsa.cval_operator, psi_0)

end_cvi = time.time()
time_cvi = end_cvi - start_cvi 



# ============== Computation Time : VFI ================= #
print ("")
print ("VFI in progress")

start_vfi = time.time()

jsa_vfi = Job_Search_adaptive()

# compute the value function via VFI
v_0 = np.ones(len(jsa_vfi.grid_points_vfi))
v_star = jsa_vfi.compute_fixed_point(jsa_vfi.Bellman_operator,
                                     v_0)

end_vfi = time.time()
time_vfi = end_vfi - start_vfi


# ========== Computation Time : CVI v.s. VFI ============ #
print ("")
print ("----------------------------------------------")
print ("")
print ("Computation time: ")
print ("")
print ("CVI : ", format(time_cvi, '.5g'), "seconds")
print ("VFI : ", 
	      int(time_vfi / 3600.), "hours", 
	      format((time_vfi/3600.- int(time_vfi/3600.))* 60, 
	      	     '.5g'), "minutes")
print ("")
print ("----------------------------------------------")
"""




"""
# This block is used to compare the graph plotted via CVI
# and VFI. Do not run this block with other ones in one 
# simulation, as there are overlapping variable names.

# ================== Plotting via CVI ==================== #

# the reservation wage
res_rule = jsa.res_rule_func(psi_star)
# the reservation utility 
res_util = jsa.res_utility(psi_star) 

psi_star_plt = psi_star.reshape((jsa.mu_size, jsa.gam_size))
res_rule_plt = res_rule.reshape((jsa.mu_size, jsa.gam_size))
res_util_plt = res_util.reshape((jsa.mu_size, jsa.gam_size))

# Plot the figure on the important part of the grid range
mu_mesh, gam_mesh = jsa.mu_mesh, jsa.gam_mesh

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_mesh, gam_mesh, 
				res_rule_plt.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('$\mu$', fontsize=15)
ax.set_ylabel('$\gamma$', fontsize=15)
ax.set_zlabel('wage', fontsize=14)
#ax.set_zlabel('Reservation utility', fontsize=14)

#ax.set_xlim((-3, 3))
#ax.set_ylim((0, 10))
ax.set_zlim((1.08, 1.65))

plt.show()


# ================== Plotting via VFI ==================== #

cvf = jsa_vfi.compute_cvf(v_star) # CVF
res_wage = jsa_vfi.res_rule_func(cvf) # reservation wage

cvf_plt = cvf.reshape((jsa_vfi.mu_size, jsa_vfi.gam_size))
res_wage_plt = res_wage.reshape((jsa_vfi.mu_size, 
	                             jsa_vfi.gam_size))

mu_mesh, gam_mesh = jsa_vfi.mu_mesh, jsa_vfi.gam_mesh

# plot the continuation value
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_mesh, gam_mesh, 
				cvf_plt.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('$\mu$', fontsize=15)
ax.set_ylabel('$\gamma$', fontsize=15)
ax.set_zlabel('continuation value', fontsize=14)
#ax.set_zlabel('wage', fontsize=14)

#ax.set_xlim((-3, 3))
#ax.set_ylim((0, 10))
#ax.set_zlim((1.08, 1.65))

plt.show()

# plot the reservation wage
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_mesh, gam_mesh, 
				res_wage.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('$\mu$', fontsize=15)
ax.set_ylabel('$\gamma$', fontsize=15)
#ax.set_zlabel('reservation wage', fontsize=14)
ax.set_zlabel('wage', fontsize=14)

#ax.set_xlim((-3, 3))
#ax.set_ylim((0, 10))
#ax.set_zlim((1.08, 1.65))

plt.show()
"""

