import autograd.numpy as np
import autograd.numpy.linalg as la

from lqr_utility import vec, mat, deblock, dlyap, dare, gain, ricc, policy_evaluation
from lqr_problems import make_lqr_objective


def value_iteration(K0, A, B, Q, X0, discount=None, min_grad_norm=None, max_iters=100):
    n = K0.size
    f, g, h = make_lqr_objective(A, B, Q, X0, discount)

    # Initialize
    P = policy_evaluation(K0, A, B, Q, discount)
    K = np.copy(K0)

    # Pre-allocate history arrays
    t_hist = np.arange(max_iters)
    x_hist = np.zeros([max_iters, n])
    f_hist = np.zeros(max_iters)
    g_hist = np.zeros([max_iters, n])
    h_hist = np.zeros([max_iters, n, n])

    # Iterate
    for i in range(max_iters-1):
        # Record history
        vK = vec(K)
        x_hist[i] = vK
        f_hist[i] = f(vK)
        g_hist[i] = g(vK)
        h_hist[i] = h(vK)
        K = gain(P, A, B, Q, discount)
        if min_grad_norm is not None:
            if la.norm(g(vec(K))) < min_grad_norm:
                # Trim off unused part of history matrices
                t_hist = t_hist[0:i+1]
                x_hist = x_hist[0:i+1]
                f_hist = f_hist[0:i+1]
                g_hist = g_hist[0:i+1]
                h_hist = h_hist[0:i+1]
                break
        P = ricc(P, A, B, Q, discount)

    # Final iterate
    K = gain(P, A, B, Q, discount)
    vK = vec(K)
    x_hist[-1] = vK
    f_hist[-1] = f(vK)
    g_hist[-1] = g(vK)
    h_hist[-1] = h(vK)
    return t_hist, x_hist, f_hist, g_hist, h_hist


def policy_iteration(K0, A, B, Q, X0, discount=None, max_iters=100, min_grad_norm=None):
    n = K0.size
    f, g, h = make_lqr_objective(A, B, Q, X0, discount)

    # Initialize
    K = np.copy(K0)
    P = policy_evaluation(K0, A, B, Q, discount)

    # Pre-allocate history arrays
    t_hist = np.arange(max_iters)
    x_hist = np.zeros([max_iters, n])
    f_hist = np.zeros(max_iters)
    g_hist = np.zeros([max_iters, n])
    h_hist = np.zeros([max_iters, n, n])

    # Iterate
    for i in range(max_iters-1):
        # Record history
        vK = vec(K)
        x_hist[i] = vK
        f_hist[i] = f(vK)
        g_hist[i] = g(vK)
        h_hist[i] = h(vK)
        if min_grad_norm is not None:
            if la.norm(g(vec(K))) < min_grad_norm:
                # Trim off unused part of history matrices
                t_hist = t_hist[0:i+1]
                x_hist = x_hist[0:i+1]
                f_hist = f_hist[0:i+1]
                g_hist = g_hist[0:i+1]
                h_hist = h_hist[0:i+1]
                break
        K = gain(P, A, B, Q, discount)
        P = policy_evaluation(K, A, B, Q, discount)

    # Final iterate
    vK = vec(K)
    x_hist[-1] = vK
    f_hist[-1] = f(vK)
    g_hist[-1] = g(vK)
    h_hist[-1] = h(vK)
    return t_hist, x_hist, f_hist, g_hist, h_hist


def riccati_direct(A, B, Q, discount=None):
    n, m = B.shape
    Qxx, Qxu, Qux, Quu = deblock(Q, n, m)
    if discount is None:
        Ad, Bd = A, B
    else:
        discount_sqrt = np.sqrt(discount)
        Ad, Bd = discount_sqrt*A, discount_sqrt*B
    P = dare(Ad, Bd, Qxx, Quu, E=None, S=Qxu)
    K = gain(P, A, B, Q, discount)
    return P, K
