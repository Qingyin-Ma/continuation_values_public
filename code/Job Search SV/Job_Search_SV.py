import numpy as np 
from scipy import interp
from numpy import maximum as npmax
from numpy import minimum as npmin
import matplotlib.pyplot as plt



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
	mc_draws : scalar(int), optional(default=10000)
		       The number of Monte Carlo samples
	"""

	def __init__(self, beta=0.95, c0_tilde=1., sig=2.5,
		         rho=.95, gam_u=1e-4, gam_xi=5e-4,
		         mu_eta=0., gam_eta=1e-6,
		         thet_min=1e-3, thet_max=10., thet_size=300,
		         mc_draws=1000):
	    self.beta, self.c0_tilde, self.sig = beta, c0_tilde, sig 
	    self.rho, self.gam_u, self.gam_xi = rho, gam_u, gam_xi
	    self.mu_eta, self.gam_eta = mu_eta, gam_eta
	    self.thet_min, self.thet_max = thet_min, thet_max
	    self.thet_size = thet_size
	    self.grid_points = np.linspace(self.thet_min, self.thet_max,
	    	                           self.thet_size)
	    self.mc_draws = mc_draws
	    self.draws = np.random.randn(self.mc_draws)


	def utility_func(self, x):
		"""
		The utility function (CRRA).
		"""
		sig = self.sig
		if sig == 1.:
			uw = np.log(x)
		else:
			uw = x**(1. - sig) / (1. - sig)
		return uw


	def cvals_operator(self, psi):
		"""
		The continuation value operator.
		"""
		c0_tilde, beta, rho = self.c0_tilde, self.beta, self.rho
		mu_eta, gam_eta = self.mu_eta, self.gam_eta
		gam_u, gam_xi = self.gam_u, self.gam_xi 
		draws = self.draws
		grid_points = self.grid_points
		utility_func = self.utility_func
		# interpolate to obtain a function
		psi_interp = lambda x: interp(x, grid_points, psi)
        
        # points outside the truncated state space will be
        # evaluated at the nearest points on the state space.
		def psi_f(x):
			x = npmax(npmin(x, self.thet_max), self.thet_min)
			return psi_interp(x)

		N = len(psi)
		new_psi = np.empty(N)

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

			new_psi[i] = utility_func(c0_tilde) + \
			             beta * np.mean(integrand)

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


	def res_rule(self, y):
		"""
		Compute the reservation wage.
		"""
		beta, sig = self.beta, self.sig
		if sig == 1.:
			w_bar = np.exp(y * (1. - beta))
		else:
			w_bar = (y*(1.-sig)*(1.-beta))**(1./ (1. - sig))

		return w_bar



jssv = Job_Search_SV(sig=1, rho=-1)

# compute the fixed point(continuation value)
N = len(jssv.grid_points)
psi_0 = np.ones(N) # initial guess of the solution
psi_star = jssv.compute_fixed_point(jssv.cvals_operator, psi_0)
res_wage = jssv.res_rule(psi_star) # the reservation wage

# plot the reservation wage
fig, ax = plt.subplots(figsize=(9,7))
ax.plot(jssv.grid_points, res_wage, linewidth=2, color='b')

# ax.set_xlim(0,10)
ax.set_ylim(0,50)

ax.set_xlabel('$\\theta$', fontsize=15)
ax.set_ylabel('wage', fontsize=14)

plt.show()















