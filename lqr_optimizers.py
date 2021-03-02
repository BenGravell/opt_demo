import autograd.numpy as np
import autograd.numpy.linalg as la

from lqr_utility import vec, mat, dlyap, dare, gain, calc_AK, calc_QK
from lqr_problems import make_lqr_objective


def policy_evaluation(K, A, B, Q):
    AK = calc_AK(K, A, B)
    QK = calc_QK(K, Q)
    PK = dlyap(AK.T, QK)
    return PK


def ricc(P, A, B, Q):
    n, m = B.shape
    AB = np.hstack([A, B])
    H = np.dot(AB.T, np.dot(P, AB)) + Q
    Hxx = H[0:n, 0:n]
    Hxu = H[0:n, n:n+m]
    Hux = H[n:n+m, 0:n]
    Huu = H[n:n+m, n:n+m]
    return Hxx - np.dot(Hxu, la.solve(Huu, Hux))


def value_iteration(K0, A, B, Q, X0, min_grad_norm=None, max_iters=100):
    n = K0.size
    f, g, h = make_lqr_objective(A, B, Q, X0)

    # Initialize
    P = policy_evaluation(K0, A, B, Q)
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
        K = gain(P, A, B, Q)
        if min_grad_norm is not None:
            if la.norm(g(vec(K))) < min_grad_norm:
                # Trim off unused part of history matrices
                t_hist = t_hist[0:i+1]
                x_hist = x_hist[0:i+1]
                f_hist = f_hist[0:i+1]
                g_hist = g_hist[0:i+1]
                h_hist = h_hist[0:i+1]
                break
        P = ricc(P, A, B, Q)

    # Final iterate
    K = gain(P, A, B, Q)
    vK = vec(K)
    x_hist[-1] = vK
    f_hist[-1] = f(vK)
    g_hist[-1] = g(vK)
    h_hist[-1] = h(vK)
    return t_hist, x_hist, f_hist, g_hist, h_hist


def policy_iteration(K0, A, B, Q, X0, max_iters=100, min_grad_norm=None):
    n = K0.size
    f, g, h = make_lqr_objective(A, B, Q, X0)

    # Initialize
    K = np.copy(K0)
    P = policy_evaluation(K0, A, B, Q)

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
        K = gain(P, A, B, Q)
        P = policy_evaluation(K, A, B, Q)

    # Final iterate
    vK = vec(K)
    x_hist[-1] = vK
    f_hist[-1] = f(vK)
    g_hist[-1] = g(vK)
    h_hist[-1] = h(vK)
    return t_hist, x_hist, f_hist, g_hist, h_hist


def riccati_direct(A, B, Q):
    n, m = B.shape
    Qxx = Q[0:n, 0:n]
    Quu = Q[n:n+m, n:n+m]
    Qxu = Q[0:n, n:n+m]
    P = dare(A, B, Qxx, Quu, E=None, S=Qxu)
    K = gain(P, A, B, Q)
    return P, K