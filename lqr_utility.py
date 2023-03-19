from copy import copy
import autograd.numpy as np
import autograd.numpy.linalg as la
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are


def quadratic_formula(a, b, c):
    """Solve the quadratic equation 0 = a*x**2 + b*x + c using the quadratic formula."""
    if a == 0:
        return [-c / b, np.nan]
    disc = b**2-4 * a * c
    disc_sqrt = disc**0.5
    den = 2 * a
    roots = [(-b+disc_sqrt) / den, (-b-disc_sqrt) / den]
    return roots


def vec(A):
    """Return the vectorized matrix A by stacking its columns."""
    return np.reshape(A, -1, order="F")


def mat(v, shape=None):
    """Return matricization i.e. the inverse operation of vec of vector v."""
    if shape is None:
        dim = int(np.sqrt(v.size))
        shape = dim, dim
    matrix = np.reshape(v, (shape[1], shape[0])).T
    return matrix


def specrad(A):
    """Spectral radius of matrix A."""
    return np.max(np.abs(la.eig(A)[0]))


# def approx_specrad(A, power=100):
#     """Approximate spectral radius of matrix A using matrix powers."""
#     return np.power(la.norm(la.matrix_power(A, power)), 1.0/power)


def sympart(A):
    """Return the symmetric part of matrix A."""
    return 0.5*(A+A.T)


def is_pos_def(A):
    """Check if matrix A is positive definite."""
    try:
        la.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def psdpart(X):
    """Return the positive semidefinite part of a symmetric matrix."""
    X = sympart(X)
    Y = np.zeros_like(X)
    eigvals, eigvecs = la.eig(X)
    for i in range(X.shape[0]):
        if eigvals[i] > 0:
            Y += eigvals[i]*np.outer(eigvecs[:, i], eigvecs[:, i])
    Y = sympart(Y)
    return Y


def posdefify(A, eps):
    E, V = la.eigh(A)
    return np.dot(V, np.dot(np.diag(np.maximum(E, eps)), V.T))


def dlyap(A, Q):
    """
    Solve the discrete-time Lyapunov equation.
    Wrapper around scipy.linalg.solve_discrete_lyapunov.
    Pass a copy of input matrices to protect them from modification.
    """
    try:
        return solve_discrete_lyapunov(np.copy(A), np.copy(Q))
    except ValueError:
        return np.full_like(Q, np.inf)


def dare(A, B, Q, R, E=None, S=None):
    """
    Solve the discrete-time algebraic Riccati equation.
    Wrapper around scipy.linalg.solve_discrete_are.
    Pass a copy of input matrices to protect them from modification.
    """
    return solve_discrete_are(copy(A), copy(B), copy(Q), copy(R), copy(E), copy(S))


def stack_AB(A, B, discount=None):
    AB = np.hstack([A, B])
    if discount is None:
        return AB
    else:
        # if not 0 <= discount <= 1:
        #     raise ValueError('Discount factor must be between 0 and 1!')
        return np.sqrt(discount)*AB


def stack_IK(K):
    m, n = K.shape
    return np.vstack([np.eye(n), K])


def calc_AK(K, A, B, discount=None):
    AB = stack_AB(A, B, discount)
    IK = stack_IK(K)
    return np.dot(AB, IK)


def calc_QK(K, Q):
    IK = stack_IK(K)
    return np.dot(IK.T, np.dot(Q, IK))


def calc_vPK(K, A, B, Q, discount=None):
    n, m = B.shape
    AK = calc_AK(K, A, B, discount)
    QK = calc_QK(K, Q)
    vQK = vec(QK)
    return la.solve(np.eye(n*n) - np.kron(AK.T, AK.T), vQK)


def calc_PK(K, A, B, Q, discount=None):
    return mat(calc_vPK(K, A, B, Q, discount))


def calc_CK(vK, A, B, Q, X0, discount=None):
    n, m = B.shape
    K = mat(vK, shape=(m, n))
    vPK = calc_vPK(K, A, B, Q, discount)
    vX0 = vec(X0)
    return np.dot(vPK, vX0)


def qfun(P, A, B, Q, discount=None):
    AB = stack_AB(A, B, discount)
    H = Q + np.dot(AB.T, np.dot(P, AB))
    return H


def deblock(H, n, m):
    Hxx = H[0:n, 0:n]
    Hxu = H[0:n, n:n+m]
    Hux = H[n:n+m, 0:n]
    Huu = H[n:n+m, n:n+m]
    return Hxx, Hxu, Hux, Huu


def qfun_comps(P, A, B, Q, discount=None):
    n, m = B.shape
    H = qfun(P, A, B, Q, discount)
    return deblock(H, n, m)


def schur_complement(H, n, m):
    Hxx, Hxu, Hux, Huu = deblock(H, n, m)
    return Hxx - np.dot(Hxu, la.solve(Huu, Hux))


def ricc(P, A, B, Q, discount=None):
    n, m = B.shape
    H = qfun(P, A, B, Q, discount)
    return schur_complement(H, n, m)


def gain(P, A, B, Q, discount=None):
    Hxx, Hxu, Hux, Huu = qfun_comps(P, A, B, Q, discount)
    K = -la.solve(Huu, Hux)
    return K


def policy_evaluation(K, A, B, Q, discount=None):
    AK = calc_AK(K, A, B, discount)
    QK = calc_QK(K, Q)
    PK = dlyap(AK.T, QK)
    return PK


def check_are(K, A, B, Q, discount=None, verbose=True):
    PK = calc_PK(K, A, B, Q, discount)
    LHS = PK
    RHS = ricc(PK, A, B, Q, discount)
    diff = la.norm(LHS-RHS)
    if verbose:
        print(' Left-hand side of the ARE: Positive definite = %s' % is_pos_def(LHS))
        print(LHS)
        print('')
        print('Right-hand side of the ARE: Positive definite = %s' % is_pos_def(RHS))
        print(RHS)
        print('')
        print('Difference')
        print(LHS-RHS)
        print('\n')
    return diff


# Compute LQR objective and/or its derivatives manually
def calc_manual(K, A, B, Q, X0, E=None, discount=1.0, derivative='cost'):
    AK = calc_AK(K, A, B, discount)
    QK = calc_QK(K, Q)
    PK = dlyap(AK.T, QK)

    if derivative == 'cost':
        CK = np.trace(np.dot(PK, X0))
        return CK
    elif derivative in ['gradient', 'hessian']:
        Hxx, Hxu, Hux, Huu = qfun_comps(PK, A, B, Q, discount)
        XK = dlyap(AK, X0)
        if derivative == 'gradient':
            EK = np.dot(Huu, K) - Hux
            G = -2*np.dot(EK, XK)
            return G
        elif derivative == 'hessian':
            SK_E_half = np.dot(E.T, Hux + np.dot(Huu, K))
            SK_E = SK_E_half + SK_E_half.T
            PKdE = dlyap(AK.T, SK_E)
            hess_EE = 2*np.trace(np.dot(np.dot(np.dot(Huu, E) + 2*(np.dot(np.sqrt(discount)*B.T, np.dot(PKdE, AK))), XK).T, E))
            return hess_EE
    else:
        raise ValueError


# Convenience aliases
def calc_cost_manual(K, A, B, Q, X0, discount=None):
    return calc_manual(K, A, B, Q, X0, E=None, discount=discount, derivative='cost')


def calc_grad_manual(K, A, B, Q, X0, discount=None):
    return calc_manual(K, A, B, Q, X0, E=None, discount=discount, derivative='gradient')


def calc_hess_manual(K, E, A, B, Q, X0, discount=None):
    return calc_manual(K, A, B, Q, X0, E=E, discount=discount, derivative='hessian')
