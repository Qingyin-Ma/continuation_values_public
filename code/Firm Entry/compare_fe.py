import numpy as np
from numpy import maximum as npmax
from numpy import minimum as npmin
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import lognorm
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import time


class Firm_Entry(object):
	"""
	The firm entry model of Fajgelbaum etc. (2016)

	x = theta + eps_x, eps_x ~ N(0, gam_x)
	where: 
		  x: output, not observed when the firm makes decisions 
		   	 at the beginning of each period.

	y = theta + eps_y, eps_y ~ N(0, gam_y)
	where: 
		  y: a public signal, observed after the firm makes 
		     decision of whether to invest or not.


	The Bayesian updating process:
	theta ~ N(mu, gam) : prior
	theta|y ~ N(mu', gam') : posterior after observing public 
							 signal y
	where: 
		  gam' = 1 / (1/gam + 1/gam_y)
		  mu' = gam'*(mu/gam + y/gam_y)


	The firm has constant absolute risk aversion:
	u(x) = (1/a) * (1 - exp(-a*x))
	where: a is the coefficient of absolute risk aversion


	f ~ h(f) = LN(mu_f, gam_f) : the entry cost / the investment cost


	The value function:
	V(f,mu,gam) = max{ E[u(x)|mu,gam]-f, beta*E[V(f',mu',gam')|mu,gam] }
	where: 
		   RHS 1st: the value of entering the market and investing
		   RHS 2nd: the expectated value of waiting


	Parameters
	----------
	beta : scalar(float), optional(default=0.95)
	       The discount factor
	a : scalar(float), optional(default=0.2)
	    The coefficient of absolute risk aversion
	mu_min : scalar(float), optional(default=-2.)
	         The minimum grid of mu
	mu_max : scalar(float), optional(default=10.)
	         The maximum grid for mu
	mu_size : scalar(int), optional(default=200)
	          The number of grid points over mu
	gam_min : scalar(float), optional(default=1e-4)
	          The minimum grid for gam
	gam_max : scalar(float), optional(default=10)
	          The maximum grid for gam
	gam_size : scalar(int), optional(default=100)
	           The number of grid points over gam
	mu_f : scalar(float), optional(default=0.)
		   The mean of the cost distribution 
		   {f_t} ~ h(f) = LN(mu_f, gam_f)
	gam_f : scalar(float), optional(default=0.01)
			The variance of the cost distribution 
			{f_t} ~ h(f) = LN(mu_f, gam_f)  
	gam_x : scalar(float), optional(default=0.1)
	   		The variance of eps_x, eps_x ~ N(0, gam_x)
	gam_y : scalar(float), optional(default=0.05)
			The variance of eps_y, eps_y ~ N(0, gam_y)
	mc_size : scalar(int), optional(default=1000)
		      The number of Monte Carlo samples 
	"""

	def __init__(self, beta=0.95, a=0.2,
		         mu_min=-2., mu_max=10., mu_size=10,
		         gam_min=1e-4, gam_max=1., gam_size=10,
		         f_min=1e-4, f_max=1., f_size=10,
		         mu_f=0., gam_f=.01, gam_x=0.1, gam_y=0.05,
		         mc_size=1000):

		self.beta, self.a = beta, a
		self.mu_f, self.gam_f = mu_f, gam_f
		self.gam_x, self.gam_y = gam_x, gam_y
		# make grids for mu
		self.mu_min, self.mu_max = mu_min, mu_max
		self.mu_size = mu_size
		self.mu_grids = np.linspace(self.mu_min, self.mu_max,
			                        self.mu_size)
		# make grids for gamma
		self.gam_min, self.gam_max = gam_min, gam_max
		self.gam_size = gam_size
		self.gam_grids = np.linspace(self.gam_min, self.gam_max,
			                         self.gam_size)
		# make grid for f
		self.f_min, self.f_max = f_min, f_max
		self.f_size = f_size
		self.f_grids = np.linspace(self.f_min, self.f_max,
			                       self.f_size)

		# make grids for CVI
		self.mu_mesh, self.gam_mesh = np.meshgrid(self.mu_grids,
			                                      self.gam_grids)
		self.grid_points = np.column_stack((self.mu_mesh.ravel(1), 
			                                self.gam_mesh.ravel(1)))
		# make grids for VFI
		self.f_mesh_vfi, self.mu_mesh_vfi, self.gam_mesh_vfi = \
		    np.meshgrid(self.f_grids, self.mu_grids, self.gam_grids)
		self.grid_points_vfi = \
		    np.column_stack((self.f_mesh_vfi.ravel(1),
		    	             self.mu_mesh_vfi.ravel(1),
		    	             self.gam_mesh_vfi.ravel(1)))
		# initial Monte Carlo draws
		self.mc_size = mc_size
		self.draws = np.random.randn(self.mc_size)


	def r(self, x, y, z):
		"""
		The exit payoff function. The expected reward of
		paying the cost f and entering the market.
		r(f, mu, gamma)
		"""
		a, gam_x = self.a, self.gam_x
		part_1 = -a * y + (a**2) * (z + gam_x) / 2.
		return (1. - np.exp(part_1)) / a - x


	def Bellman_operaotr(self, v):
		"""
		The Bellman operator.
		"""
		beta = self.beta
		gam_y = self.gam_y
		f_min, f_max = self.f_min, self.f_max
		mu_min, mu_max = self.mu_min, self.mu_max
		gam_min, gam_max = self.gam_min, self.gam_max
		mu_f, gam_f = self.mu_f, self.gam_f
		mc_size, draws = self.mc_size, self.draws
		grid_points_vfi = self.grid_points_vfi

		# interpolate to obtain a function
		v_interp = LinearNDInterpolator(grid_points_vfi, v)
		
		def v_f(x, y, z):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			"""
			x = npmin(npmax(x, f_min), f_max)
			y = npmin(npmax(y, mu_min), mu_max)
			z = npmin(npmax(z, gam_min), gam_max)
			return v_interp(x, y, z)

		N = len(v)
		new_v = np.empty(N)

		for i in range(N):
			f, mu, gam = grid_points_vfi[i, :]
			# MC draws for y
			y_draws = mu + np.sqrt(gam + gam_y)* draws
			
			# MC draws for f'
			f_prime = np.exp(mu_f + np.sqrt(gam_f) * draws)
			# MC draws for gamma'
			gam_prime = 1. / (1. / gam + 1. / gam_y)
			# MC draws for mu'
			mu_prime = gam_prime * (mu / gam + y_draws / gam_y)

			# MC draws for v(f',mu',gamma')
			integrand = v_f(f_prime, mu_prime, 
				            gam_prime* np.ones(mc_size))
			
			# CVF
			cvf = beta * np.mean(integrand)
			# exit payoff
			exit_payoff = self.r(f, mu, gam)

			new_v[i] = max(exit_payoff, cvf)

		return new_v


	def compute_cvf(self, v):
		"""
		Compute the continuation value based on the 
		value function.
		"""
		beta = self.beta
		gam_y = self.gam_y
		f_min, f_max = self.f_min, self.f_max
		mu_min, mu_max = self.mu_min, self.mu_max
		gam_min, gam_max = self.gam_min, self.gam_max
		mu_f, gam_f = self.mu_f, self.gam_f
		mc_size, draws = self.mc_size, self.draws
		grid_points_vfi = self.grid_points_vfi
		grid_points = self.grid_points

		# interpolate to obtain a function
		v_interp = LinearNDInterpolator(grid_points_vfi, v)

		def v_f(x, y, z):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			"""
			x = npmin(npmax(x, f_min), f_max)
			y = npmin(npmax(y, mu_min), mu_max)
			z = npmin(npmax(z, gam_min), gam_max)
			return v_interp(x, y, z)

		N = len(grid_points)
		cvf = np.empty(N)

		for i in range(N):
			mu, gam = grid_points[i, :]
			# MC draws for y
			y_draws = mu + np.sqrt(gam + gam_y)* draws
			
			# MC draws for f'
			f_prime = np.exp(mu_f + np.sqrt(gam_f) * draws)
			# MC draws for gamma'
			gam_prime = 1. / (1. / gam + 1. / gam_y)
			# MC draws for mu'
			mu_prime = gam_prime * (mu / gam + y_draws / gam_y)

			# MC draws for v(f',mu',gamma')
			integrand = v_f(f_prime, mu_prime, 
				            gam_prime* np.ones(mc_size))
			
			# CVF
			cvf[i] = beta * np.mean(integrand)

		return cvf


	def cvals_operator(self, psi):
		"""
		The continuation value operator
		--------------------------------
		Qpsi(mu,gam) 
		= beta * integral( 
			           max{reward,phi(mu',gam')} 
			           * h(f')*l(y|mu,gam) ) 
                 d(f',y)
		where:
			  f ~ h(f) = LN(mu_f, gam_f) : the entry/investment cost
			  gam' = 1/(1/gam + 1/gam_y)
			  mu' = gam' * (mu/gam + y/gam_y)
			  reward = (1/a) - (1/a)*exp[-a*mu' + (a**2)*(gam'+gam_x)/2] - f'
   			  l(y|mu, gam) = N(mu, gam + gam_y)


		The operator Q is a contraction mapping on 
		(b_{ell} Y, rho_{ell}) with unique fixed point psi^*.


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
		beta = self.beta
		gam_y = self.gam_y
		mu_min, mu_max = self.mu_min, self.mu_max
		gam_min, gam_max = self.gam_min, self.gam_max
		mu_f, gam_f = self.mu_f, self.gam_f
		mc_size, draws = self.mc_size, self.draws
		grid_points = self.grid_points
		psi_interp = LinearNDInterpolator(grid_points, psi)

		def psi_f(x, y):
			"""
			Interpolate but extrapolate using the nearest value
			on the grid.
			Notice that the arguments of this function 
			are ordered by : psi_f(mu, gamma).
			"""
			x = npmin(npmax(x, mu_min), mu_max)
			y = npmin(npmax(y, gam_min), gam_max)
			return psi_interp(x, y)

		N = len(psi)
		new_psi = np.empty(N)

		for i in range(N):
			mu, gam = grid_points[i, :]
			# MC draws for y
			y_draws = mu + np.sqrt(gam + gam_y)* draws
			# MC draws for f'
			f_prime = np.exp(mu_f + np.sqrt(gam_f) * draws)

			# MC draws for gamma'
			gam_prime = 1. / (1. / gam + 1. / gam_y)
			# MC draws for mu'
			mu_prime = gam_prime * (mu / gam + y_draws / gam_y)

			# MC draws for r(f',mu',gamma')
			rprime_draws = self.r(f_prime, mu_prime, gam_prime)
			# MC draws for psi(mu',gamma')
			psiprime_draws = psi_f(mu_prime,
				                   gam_prime * np.ones(mc_size))
			# MC draws: max{r(f',mu',gam'), psi(mu',gam')}
			integrand = npmax(rprime_draws, psiprime_draws)
			new_psi[i] = beta * np.mean(integrand)

		return new_psi


	def compute_fixed_point(self, Q, psi, error_tol=1e-3,
		                    max_iter=500, verbose=1):
		"""
		Compute the fixed point.
		"""
		iteration = 0
		error = error_tol + 1.

		while iteration < max_iter and error > error_tol:
			Qpsi = Q(psi)
			error = max(abs(Qpsi - psi))
			psi = Qpsi
			iteration += 1

			if verbose:
				print ("Computing iteration", iteration," with error", error)

		return psi


	def res_rule(self, psi):
		"""
		The reservation cost function.
		"""
		a, gam_x = self.a, self.gam_x
		grid_points = self.grid_points
		part_1 = -a * grid_points[:,0] + \
		         (a**2)* (grid_points[:,1] + gam_x) / 2.
		part_2 = (1. - np.exp(part_1)) / a

		return part_2 - psi



# ================= Computation time: CVI ================= #

print ("")
print ("CVI in progress ...")

start_cvi = time.time()

fe = Firm_Entry()

# compute fixed point via compute_fixed_point
psi_0 = np.ones(len(fe.grid_points))
psi_star = fe.compute_fixed_point(fe.cvals_operator, psi_0)

end_cvi = time.time()
time_cvi = end_cvi - start_cvi 



# ================= Computation time: VFI ================= #

print ("")
print ("VFI in progress ...")

start_vfi = time.time()

fe_vfi = Firm_Entry()

# compute the fixed point via VFI
v0 = np.ones(len(fe_vfi.grid_points_vfi))
v_star = fe_vfi.compute_fixed_point(fe.Bellman_operaotr, v0)

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





# ================== Plot results: VFI ==================== #

# compute the continuation value
cvf = fe_vfi.compute_cvf(v_star)
# compute the reservation cost
res_cost_vfi = fe_vfi.res_rule(cvf)
# compute the probability of investment
prob_inv_vfi = lognorm.cdf(res_cost_vfi, 
	                       s=np.sqrt(fe_vfi.gam_f),
	                       scale=np.exp(fe_vfi.mu_f))


cvf_plt = cvf.reshape((fe_vfi.mu_size, fe_vfi.gam_size))
res_cost_vfi_plt = res_cost_vfi.reshape((fe_vfi.mu_size,
	                                     fe_vfi.gam_size))
prob_inv_vfi_plt = prob_inv_vfi.reshape((fe_vfi.mu_size,
	                                     fe_vfi.gam_size))


mu_mesh, gam_mesh = fe_vfi.mu_mesh, fe_vfi.gam_mesh

# plot the perceived probability of investment 
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_mesh, gam_mesh, 
				prob_inv_vfi_plt.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('$\mu$', fontsize=15)
ax.set_ylabel('$\gamma$', fontsize=15)
ax.set_zlabel('probability', fontsize=14)

"""
# plot the reservation cost
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_mesh, gam_mesh, 
				res_cost_vfi_plt.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('$\mu$', fontsize=15)
ax.set_ylabel('$\gamma$', fontsize=15)
ax.set_zlabel('cost', fontsize=14)

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
"""





# ================== Plot results: CVI ==================== #

# compute the reservation cost
res_cost = fe.res_rule(psi_star)
# compute the probability of investment
prob_inv = lognorm.cdf(res_cost, s=np.sqrt(fe.gam_f),
	                   scale=np.exp(fe.mu_f))


psi_star_plt = psi_star.reshape((fe.mu_size, fe.gam_size))
res_cost_plt = res_cost.reshape((fe.mu_size, fe.gam_size))
prob_inv_plt = prob_inv.reshape((fe.mu_size, fe.gam_size))


mu_mesh, gam_mesh = fe.mu_mesh, fe.gam_mesh


# plot the perceived probability of investment 
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_mesh, gam_mesh, 
				prob_inv_plt.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('$\mu$', fontsize=15)
ax.set_ylabel('$\gamma$', fontsize=15)
ax.set_zlabel('probability', fontsize=14)

"""
# plot the reservation cost
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_mesh, gam_mesh, 
				res_cost_plt.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('$\mu$', fontsize=15)
ax.set_ylabel('$\gamma$', fontsize=15)
ax.set_zlabel('cost', fontsize=14)

# plot the continuation value
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(mu_mesh, gam_mesh, 
				psi_star_plt.T,
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

ax.set_xlabel('$\mu$', fontsize=15)
ax.set_ylabel('$\gamma$', fontsize=15)
ax.set_zlabel('continuation value', fontsize=14)
"""

plt.show()



