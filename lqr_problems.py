import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as npr
from autograd import grad, hessian
from functools import partial

from lqr_utility import vec, mat, specrad, calc_CK, posdefify, is_pos_def, calc_AK


def gen_rand_pd(n, round_places=None):
    Qdiag = np.diag(npr.rand(n))
    Qvecs = la.qr(npr.randn(n, n))[0]
    Q = np.dot(Qvecs, np.dot(Qdiag, Qvecs.T))
    if round_places is not None:
        Q = Q.round(round_places)
        if not is_pos_def(Q):
            Q = Q + (10**(-round_places))*np.eye(n)
    return Q


def gen_rand_A(n, rho=None, seed=None, round_places=None):
    npr.seed(seed)
    A = npr.randn(n, n)
    if rho is not None:
        A = A * (rho / specrad(A))
    if round_places is not None:
        A = A.round(round_places)
    return A


def gen_rand_B(n, m, seed=None, round_places=None):
    npr.seed(seed)
    B = npr.rand(n, m)
    if round_places is not None:
        B = B.round(round_places)
    return B


def gen_lqr_problem(n, m, rho=None, round_places=None, seed=None):
    A = gen_rand_A(n, rho, seed, round_places)
    B = gen_rand_B(n, m, seed+1, round_places)
    Q = gen_rand_pd(n+m, round_places)
    X0 = gen_rand_pd(n, round_places)
    return A, B, Q, X0


def make_lqr_objective(A, B, Q, X0, discount=1.0):
    if A.shape[0] != B.shape[0] != X0.shape[0] or np.sum(B.shape) != Q.shape[0]:
        raise ValueError('Incompatible dimensions!')

    # Create the LQR objective and get derivatives
    fs = partial(calc_CK, A=A, B=B, Q=Q, X0=X0, discount=discount)
    g = grad(fs)
    h = hessian(fs)

    # Make an enhanced version of the objective which properly handles unstable A+BK
    # Only tries to solve dlyap if A+BK is stable, else returns inf
    # This is necessary for line search
    # Also keeps ill-defined situations out of the derivatives when using autograd
    # by modding f **after** calling grad and hessian - otherwise specrad() messes up autograd
    def fus(vK, n, m):
        K = mat(vK, shape=(m, n))
        AK = calc_AK(K, A, B, discount=1.0)
        if specrad(AK) < 1.0:
            return fs(vK)
        else:
            return np.inf
    n, m = B.shape
    f = partial(fus, n=n, m=m)
    return f, g, h
