import autograd.numpy as np
import autograd.numpy.linalg as la
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are
from copy import copy


class PrintColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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


def gain(P, A, B, Q):
    n, m = B.shape
    AB = np.hstack([A, B])
    H = np.dot(AB.T, np.dot(P, AB)) + Q
    Hux = H[n:n+m, 0:n]
    Huu = H[n:n+m, n:n+m]
    K = -la.solve(Huu, Hux)
    return K


def calc_AK(K, A, B):
    return A + np.dot(B, K)


def calc_QK(K, Q):
    m, n = K.shape
    IK = np.vstack([np.eye(n), K])
    return np.dot(IK.T, np.dot(Q, IK))


def calc_vPK(K, A, B, Q):
    n, m = B.shape
    AK = calc_AK(K, A, B)
    QK = calc_QK(K, Q)
    vQK = vec(QK)
    return la.solve(np.eye(n*n) - np.kron(AK.T, AK.T), vQK)


def calc_CK(vK, A, B, Q, X0):
    n, m = B.shape
    K = mat(vK, shape=(m, n))
    vPK = calc_vPK(K, A, B, Q)
    vX0 = vec(X0)
    return np.dot(vPK, vX0)


def check_are(K, A, B, Q, verbose=True):
    n, m = B.shape
    AB = np.hstack([A, B])
    PK = mat(calc_vPK(K, A, B, Q))
    H = np.dot(AB.T, np.dot(PK, AB)) + Q
    Hxx = H[0:n, 0:n]
    Huu = H[n:n+m, n:n+m]
    Hux = H[n:n+m, 0:n]
    LHS = PK
    RHS = Hxx-np.dot(Hux.T, la.solve(Huu, Hux))
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


def calc_cost_manual(K, A, B, Q, X0):
    AK = calc_AK(K, A, B)
    QK = calc_QK(K, Q)
    PK = dlyap(AK.T, QK)
    CK = np.trace(np.dot(PK, X0))
    return CK


def calc_grad_manual(K, A, B, Q, X0):
    n, m = B.shape
    AK = calc_AK(K, A, B)
    QK = calc_QK(K, Q)
    PK = dlyap(AK.T, QK)
    XK = dlyap(AK, X0)

    AB = np.hstack([A, B])
    H = np.dot(AB.T, np.dot(PK, AB))+Q
    Hux = H[n:n+m, 0:n]
    Huu = H[n:n+m, n:n+m]

    EK = np.dot(Huu, K)-Hux
    G = -2 * np.dot(EK, XK)
    return G


def calc_hess_manual(K, E, A, B, Q, X0):
    n, m = B.shape
    AK = calc_AK(K, A, B)
    QK = calc_QK(K, Q)
    PK = dlyap(AK.T, QK)
    XK = dlyap(AK, X0)

    AB = np.hstack([A, B])
    H = np.dot(AB.T, np.dot(PK, AB))+Q
    Hux = H[n:n+m, 0:n]
    Huu = H[n:n+m, n:n+m]

    SK_E_half = np.dot(E.T, Hux+np.dot(Huu, K))
    SK_E = SK_E_half+SK_E_half.T
    PKdE = dlyap(AK.T, SK_E)
    hess_EE = 2 * np.trace(np.dot(np.dot(np.dot(Huu, E)+2 * (np.dot(B.T, np.dot(PKdE, AK))), XK).T, E))
    return hess_EE
