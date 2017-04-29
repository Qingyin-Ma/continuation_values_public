import numpy as np
from scipy.interpolate import LinearNDInterpolator
#from scipy import interp
from numpy import maximum as npmax 
from numpy import minimum as npmin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Firm_Exit_multi(object):
	"""
	Hopenhayn (1992): The firm decides whether or not to exit
					  the market in the next period.
	
	IMPORTANT :	
	This program is a parametric class problem.
	The state space is two dimensional, with state variables
	$Z_t$ and $c_f$ (a class fixed values).

	The state process is AR(1):
	Z_{t+1} = rho Z_t + b + eps_{t+1}, eps_t ~ N(0, sigma^2)

	The value function:
	v^*(z) = max {r(z), 
				  c(z) + beta int v^*(z') f(z'|z) dz'}
	where:
	 f(z'|z) = N(rho z + b, sigma^2)

	 The exit payoff function :
	 r(z) = G e^{1/(1-alpha) z} - c_f

	 The flow continuation payoff :
	 c(z) = r(z) = G e^{1/(1-alpha) z} - c_f

	 Note: G is a constant determined by model primitives.
	 	The firm' output q(a,l) = a l^{alpha}
	 	l^{*} = argmax {p * q(a,l) - w * l}
	 	r(z) = c(z) and G is obtained by using l^{*} as the 
	 	labor input.


	Parameters
	-----------
	beta : scalar(float), optional(default=0.95)
		  the discount factor
	c_f : scalar(float), optional(default=10.0)
		  the fixed cost of staying in the industry
	rho : scalar(float), optional(default=0.6)
		  the autoregression coefficient of the 
		  state process (Z_t)_{t geq 0}
	b : scalar(float), optional(default=1.0)
		the drift constant of the state prcess 
		(Z_t)_{t geq 0}
	sigma : scalar(float), optional(default=1.0)
			the variance of the stochastic shock of
			of the state process  
	alpha : scalar(float), optional(default=0.5)
			the labor factor of the production function
	p, w : scalars(float), optional(default=0.02, 0.04)
		   the output and input prices
	z_min, z_max : scalars(float), optional(default=-6.0, 6.0)
				   minimum and maximum grid values
	grid_size : scalar(int), optional(default=200)
				the grid size
	mc_size : scalar(int), optional(default=1000)
			  the number of Monte Carlo draws used
			  to approximate the integration.
	"""

	def __init__(self, beta=0.95, alpha=0.5, p=0.15, w=0.15,
				  mc_size=1000, rho=0.7, b=0.0, sigma=1.0,
				  cf_min=0.0, cf_max=10.0, cf_grid_size=100,
				  z_min=-10.0, z_max=10.0, z_grid_size=200
				  ):
		
		self.beta, self.alpha = beta, alpha
		self.p, self.w = p, w 
		self.rho, self.b, self.sigma = rho, b, sigma
		# making grids for cf
		self.cf_min, self.cf_max = cf_min, cf_max
		self.cf_grid_size = cf_grid_size
		self.cf_grids = np.linspace(self.cf_min, self.cf_max,
									self.cf_grid_size)
		# making grids for Z
		self.z_min, self.z_max = z_min, z_max
		self.z_grid_size = z_grid_size
		self.z_grids = np.linspace(self.z_min, self.z_max, 
								   self.z_grid_size)

		# an alternative way to make grids, uncomment below
		#grid_pos = self.makegrid(0, z_max, self.grid_size/2, 1)
		#grid_neg = np.sort(-grid_pos)
		#self.grid_points = np.concatenate((grid_neg, grid_pos))

		# all grid combinations 
		self.z_mesh, self.cf_mesh = np.meshgrid(self.z_grids, 
												self.cf_grids)
		self.grid_points = np.column_stack((self.z_mesh.ravel(1), 
										    self.cf_mesh.ravel(1)))

		self.G = (self.alpha * self.p /self.w)**(1.0/(1.0-self.alpha)) \
				  * ((1.0-self.alpha)/self.alpha) * self.w
		self.mc_size = mc_size
		self.draws = np.random.randn(self.mc_size)


	def makegrid(self, amin, amax, asize, ascale):
		"""
		Generates grid a with asize number of points ranging
		from amin to amax. 

		Parameters
		----------
		amin: the minimum grid 
		amax: the maximum grid 
		asize: the number of grid points to be generated

		ascale=1: generates equidistant grids, same as np.linspace
		ascale>1: the grid is scaled to be more dense in the 
				  lower value range

		Returns
		-------
		a : array_like(float, ndim=1, length=asize)
			The generated grid points
		
		"""
		a = np.empty(asize)
		adelta = (amax - amin) / ((asize - 1)**ascale)

		for i in range(asize):
			a[i] = amin + adelta * (i ** ascale)
		return a


	def r(self, x, y):
		"""
		The exit payoff function.
		"""
		G, alpha = self.G, self.alpha
		return G * np.exp((1.0/(1.0-alpha)) * x) - y


	def c(self, x, y):
		"""
		The flow continuation payoff function.
		"""
		G, alpha = self.G, self.alpha
		return G * np.exp((1.0/(1.0-alpha)) * x) - y


	def cval_operator(self, psi):
		"""
		The continuation value operator
		--------------------------------
		Q psi(z) 
			= c(z) + 
			  beta * int max {r(z'), psi(z')} f(z'|z) dz'
		where:
			f(z'|z) = N(rho z + b, sigma^2)


		Parameters
		-----------
		psi : array_like(float, ndim=1, length=len(grid_points))
			  An approximate fixed point represented as a one 
			  dimensional array.


		Returns
		--------
		psi_new : array_like(float, ndim=1, length=len(grid_points))
						  The updated fixed point.		
		"""
		rho, b, sigma, beta = self.rho, self.b, self.sigma, self.beta
		z_min, z_max = self.z_min, self.z_max

		# interpolate to get an approximate fixed point function
		psi_interp = LinearNDInterpolator(self.grid_points, psi) 
		N = len(psi) 
		psi_new = np.empty(N)

		for i in range(N):
			z, cf = self.grid_points[i, :]
			# sample z' from f(z'|z) = N(rho z + b, sigma^2)
			z_prime = rho * z + b + sigma * self.draws

			# samples outside the truncated state space is 
			# replaced by the nearest grids of the state space
			z_prime = npmax(npmin(z_prime, z_max), z_min)

			intgrand_1 = self.r(z_prime, cf) # r(z';c_f) samples
			intgrand_2 = psi_interp(z_prime, cf) # psi(z';c_f) samples

			# approximate integral via Monte Carlo integration 
			integral = np.mean(npmax(intgrand_1, intgrand_2))

			psi_new[i] = self.c(z, cf) + beta * integral # Q psi(z;c_f)

		return psi_new


	def compute_fixed_point(self, Q, psi, max_iter=500,
							error_tol=1e-4, verbose=1):
		"""
		Compute the fixed point of the continuation value
		operator.
		"""
		error = error_tol + 1.0
		iteration = 0

		while error > error_tol and iteration < max_iter:

			psi_new = Q(psi)
			error = np.max(abs(psi_new - psi)) 
			psi = psi_new
			iteration += 1

			if verbose:
				print ("Computing iteration ", iteration, " with error ", error)

		return psi


fe = Firm_Exit_multi()
# initial guess for the fixed point
psi_init = fe.r(fe.grid_points[:,0], fe.grid_points[:,1]) 
# compute the fixed point (continuation value function)
psi_star = fe.compute_fixed_point(fe.cval_operator, psi_init)

# an alternative way to compute the fixed point
	# (iterate a fixed number of times)
	# uncomment the next three lines to implement
# psi_star = np.ones(fe.grid_size)
# for i in range(100):
# 	psi_star = fe.cval_operator(psi_star)

# the exit payoff function
r = fe.r(fe.grid_points[:,0], fe.grid_points[:,1]) 
# the value function
v_star = npmax(r, psi_star)


fig = plt.figure()
ax = fig.gca(projection='3d')

z_mesh, cf_mesh = fe.z_mesh, fe.cf_mesh


# plot the continuation value 
psi_star_plt = psi_star.reshape((fe.z_grid_size, 
								 fe.cf_grid_size))
ax.plot_surface(z_mesh[:,60:125], cf_mesh[:,60:125], 
				psi_star_plt.T[:,60:125],
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)

"""
# plot the value function
v_star_plt = v_star.reshape((fe.z_grid_size,
							 fe.cf_grid_size))
ax.plot_surface(z_mesh[:,60:125], cf_mesh[:,60:125], 
				v_star_plt.T[:,60:125],
				rstride=2, cstride=3, cmap=cm.jet,
				alpha=0.5, linewidth=0.25)
"""

ax.set_xlim(-4, 3)
ax.set_ylim(0, 10)
ax.set_zlim(-30, 80)

ax.set_xlabel('$z$', fontsize=16)
ax.set_ylabel('$c_f$', fontsize=16)
#ax.set_zlabel('value function', fontsize=14)
ax.set_zlabel('continuation value', fontsize=14)

plt.show()






