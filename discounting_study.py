import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from time import time
import os

from lqr_utility import vec, mat, dlyap, dare, check_are, calc_AK
from lqr_problems import gen_lqr_problem, make_lqr_objective
from optimizers import Objective
from lqr_optimizers import value_iteration, policy_iteration, riccati_direct


def lsim_cl(A, B, K, T, x0=None):
    AK = calc_AK(K, A, B)
    t_hist = np.arange(T+1)
    x_hist = np.zeros([T+1, n])
    if x0 is not None:
        x_hist[0] = x0
    for t in range(T):
        x_hist[t+1] = AK @ x_hist[t]
    return t_hist, x_hist


# Reset global options
np.set_printoptions(precision=3)
seed = 1

# Make LQR problem data
A, B, Q, X0 = gen_lqr_problem(n=3, m=2, rho=1.05, round_places=1, seed=seed)
n, m = B.shape

# Solve discounted LQR problems
P_undiscounted, K_undiscounted = riccati_direct(A, B, Q, discount=None)
cmap = 'Blues_r'
N = 2
T = 50
x0 = np.ones(n)
discounts = np.linspace(0.0, 2.0, 2*N+1)
fig, ax = plt.subplots(nrows=2, ncols=2*N+1, figsize=(3*2*N, 4))
for i, discount in enumerate(discounts):
    P, K = riccati_direct(A, B, Q, discount)
    t_hist, x_hist = lsim_cl(A, B, K, T, x0)
    ax[0, i].imshow(K, vmin=K_undiscounted.min(), vmax=K_undiscounted.max(), cmap=cmap)
    ax[0, i].set_title('Discount = %.2f' % discount)
    ax[1, i].plot(t_hist, x_hist)
    ax[1, i].grid('on')
fig.tight_layout()
plt.show()
