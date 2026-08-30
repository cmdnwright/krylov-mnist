'''
- householder reduction to hessenberg (general) and tridiagonal (symmetric)
- power iteration for dominant eigenpair
- onverse iteration and rayleigh quotient iteration (RQI)
- structure aware solvers for tridiagonal / hessenberg matrices
- symmetric QR iteration (tridiagonal)
- francis implicit double shift QR iteration for real hessenberg matrices
'''

import math
from dataclasses import dataclass

import numpy as np
import numpy.linalg as la

rng = np.random.default_rng()


def normalize(v: np.ndarray) -> np.ndarray:
    '''normalize v / ||v||_2, or v if v == 0'''
    n = la.norm(v)
    return v if n == 0 else (v / n)


def rayleigh_quotient(A: np.ndarray, x: np.ndarray) -> float:
    '''rayleigh quotient x^T A x / (x^T x)'''
    x = normalize(x)
    return float(x @ (A @ x))


def householder_hessenberg(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''reduce a general real matrix A to upper hessenberg form H via
    orthogonal similarity  A = Q H Q^T

    Parameters
    ----------
    A : (n, n) ndarray
        matrix to reduce

    Returns
    -------
    H : (n, n) ndarray
        upper hessenberg matrix.
    Q : (n, n) ndarray
        orthogonal matrix with A = Q @ H @ Q.T
    '''
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    Q = np.eye(n)

    for k in range(n - 2):
        x = A[k + 1:, k]
        normx = la.norm(x)
        if normx == 0:
            continue

        sigma = 1.0 if x[0] >= 0 else -1.0
        u1 = x[0] + sigma * normx
        v = x.copy()
        v[0] = u1
        v = v / la.norm(v)

        # left H = (I - 2vv^T) H
        A[k + 1:, k:] -= 2.0 * np.outer(v, v @ A[k + 1:, k:])
        # right H <- H (I - 2vv^T)
        A[:, k + 1:] -= 2.0 * np.outer(A[:, k + 1:] @ v, v)
        # Q = Q (I - 2vv^T)
        Q[:, k + 1:] -= 2.0 * np.outer(Q[:, k + 1:] @ v, v)

    # zero out exact lower-lower entries
    H = A.copy()
    H[np.tril_indices_from(H, -2)] = 0.0
    return H, Q


def householder_tridiagonal(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''reduce a real symmetric matrix A to tridiagonal form T via
    orthogonal similarity  A = Q T Q^T

    Parameters
    ----------
    A : (n, n) ndarray, symmetric
        matrix to reduce

    Returns
    -------
    T : (n, n) ndarray
        tridiagonal matrix (up to roundoff)
    Q : (n, n) ndarray
        orthogonal matrix with A = Q @ T @ Q.T
    '''
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    Q = np.eye(n)

    for k in range(n - 2):
        x = A[k + 1:, k]
        normx = la.norm(x)
        if normx == 0:
            continue

        sigma = 1.0 if x[0] >= 0 else -1.0
        u1 = x[0] + sigma * normx
        v = x.copy()
        v[0] = u1
        v = v / la.norm(v)

        A[k + 1:, k:] -= 2.0 * np.outer(v, v @ A[k + 1:, k:])
        A[:, k + 1:] -= 2.0 * np.outer(A[:, k + 1:] @ v, v)

        Q[:, k + 1:] -= 2.0 * np.outer(Q[:, k + 1:] @ v, v)

        A = (A + A.T) / 2.0

    T = A.copy()
    i, j = np.indices(T.shape)
    T[np.abs(i - j) > 1] = 0.0
    return T, Q


def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    '''solve a tridiagonal system Tx = rhs using the thomas algorithm

    T = diag(a, -1) + diag(b, 0) + diag(c, +1), with a[0] unused and c[-1] unused

    Parameters
    ----------
    a : (n,) ndarray
        subdiagonal (a[0] ignored)
    b : (n,) ndarray
        main diagonal
    c : (n,) ndarray
        superdiagonal (c[-1] ignored)
    rhs : (n,) ndarray
        rhs vector of the system

    Returns
    -------
    x : (n,) ndarray
        solution
    '''
    a = a.copy().astype(float)
    b = b.copy().astype(float)
    c = c.copy().astype(float)
    d = rhs.copy().astype(float)

    n = len(b)

    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    x = np.zeros_like(d)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def solve_hessenberg(H: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    '''solve Hx = rhs for upper hessenberg H with O(n^2) gaussian elimination without pivoting

    Parameters
    ----------
    H : (n, n) ndarray
        upper hessenberg matrix
    rhs : (n,) ndarray
        rhs of the equation hx = rhs

    Returns
    -------
    x : (n,) ndarray
        solution
    '''
    H = H.copy().astype(float)
    b = rhs.copy().astype(float)
    n = H.shape[0]

    for k in range(n - 1):
        if abs(H[k, k]) < 1e-15:
            pivot = np.argmax(np.abs(H[k:, k])) + k
            if pivot != k:
                H[[k, pivot]] = H[[pivot, k]]
                b[k], b[pivot] = b[pivot], b[k]

        factor = H[k + 1, k] / H[k, k]
        H[k + 1, k:] -= factor * H[k, k:]
        b[k + 1] -= factor * b[k]

    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - H[i, i + 1:] @ x[i + 1:]) / H[i, i]

    return x


@dataclass
class PowerResult:
    eigenvector: np.ndarray
    eigenvalue: float
    lambdas: list[float]


def power_iteration(A: np.ndarray, v0: np.ndarray | None = None, maxit: int = 500, tol: float = 1e-10) -> PowerResult:
    '''power iteration for dominant eigenpair of A (works best if A is symmetric
    or diagonalizable with a unique dominant eigenvalue).

    Parameters
    ----------
    A : (n, n) ndarray
    v0 : (n,) ndarray or None
        Initial vector.
    maxit : int
    tol : float
        Stopping tolerance on change of Rayleigh quotient.

    Returns
    -------
    PowerResult(eigenvector, eigenvalue, lambdas)
    '''
    n = A.shape[0]
    v = rng.standard_normal(n) if v0 is None else np.array(v0, dtype=float)
    v = normalize(v)

    lambdas = []
    lam_old = rayleigh_quotient(A, v)
    lam = 0.0

    for _ in range(maxit):
        w = A @ v
        v = normalize(w)
        lam = rayleigh_quotient(A, v)
        lambdas.append(lam)

        if abs(lam - lam_old) <= tol * (1 + abs(lam)):
            break
        lam_old = lam

    return PowerResult(v, lam, lambdas)


@dataclass
class InverseIterationResult:
    eigenvector: np.ndarray
    eigenvalue: float
    lambdas: list[float]


def inverse_iteration(A: np.ndarray, mu: float, v0: np.ndarray | None = None, maxit: int = 50, tol: float = 1e-12, structure: str = 'full') -> InverseIterationResult:
    '''inverse iteration for an eigenvalue near shift mu

    Parameters
    ----------
    A : (n, n) ndarray
    mu : float
        Shift
    v0 : (n,) ndarray or None
    maxit : int
    tol : float
        Residual tolerance: ||Av - λv|| <= tol*(||A||+|λ|)
    structure : {'full', 'tridiag', 'hessenberg'}
        If tridiag or hessenberg, use specialized O(n^2) or O(n) solves

    Returns
    -------
    InverseIterationResult(eigenvector, eigenvalue, lambdas)
    '''
    A = np.array(A, dtype=float, copy=False)
    n = A.shape[0]
    v = rng.standard_normal(n) if v0 is None else np.array(v0, dtype=float)
    v = normalize(v)

    I = np.eye(n)
    lambdas = []
    lam = 0.0

    for _ in range(maxit):
        # (A - mu I) w = v
        if structure == 'full':
            w = la.solve(A - mu * I, v)
        elif structure == 'tridiag':
            # T w = v
            a = np.diag(A, k=-1).copy()
            b = np.diag(A).copy() - mu
            c = np.diag(A, k=1).copy()
            w = solve_tridiagonal(a, b, c, v)
        elif structure == 'hessenberg':
            w = solve_hessenberg(A - mu * I, v)
        else:
            raise ValueError(f'unknown structure')

        v = normalize(w)
        lam = rayleigh_quotient(A, v)
        lambdas.append(lam)

        res = la.norm(A @ v - lam * v)
        if res <= tol * (la.norm(A, 1) + abs(lam)):
            break

    return InverseIterationResult(v, lam, lambdas)


@dataclass
class RQIResult:
    eigenvector: np.ndarray
    eigenvalue: float
    mus: list[float]


def rayleigh_quotient_iteration(A: np.ndarray, v0: np.ndarray | None = None, maxit: int = 30, tol: float = 1e-12, structure: str = 'full') -> RQIResult:
    '''rayleigh quotient iteration (cubic convergence for symmetric/Hermitian A)

    Parameters
    ----------
    A : (n, n) ndarray
        target matrix
    v0 : (n,) ndarray or None
        initial guess
    maxit : int
        maximum number of iterations
    tol : float
        tolterance for residual stopping
    structure : {'full', 'tridiag', 'hessenberg'}
        structure of A used to accelerate the linear solve.

    Returns
    -------
    RQIResult(eigenvector, eigenvalue, mus)
    '''
    A = np.array(A, dtype=float, copy=False)
    n = A.shape[0]
    v = rng.standard_normal(n) if v0 is None else np.array(v0, dtype=float)
    v = normalize(v)

    I = np.eye(n)
    mu = rayleigh_quotient(A, v)
    mus = [mu]

    for _ in range(maxit):
        # Solve (A - mu I) w = v
        if structure == 'full':
            w = la.solve(A - mu * I, v)
        elif structure == 'tridiag':
            a = np.diag(A, k=-1).copy()
            b = np.diag(A).copy() - mu
            c = np.diag(A, k=1).copy()
            w = solve_tridiagonal(a, b, c, v)
        elif structure == 'hessenberg':
            w = solve_hessenberg(A - mu * I, v)
        else:
            raise ValueError(f'unknown structure ')

        v = normalize(w)
        mu_new = rayleigh_quotient(A, v)
        mus.append(mu_new)

        res = la.norm(A @ v - mu_new * v)
        if res <= tol * (la.norm(A, 1) + abs(mu_new)):
            mu = mu_new
            break

        mu = mu_new

    return RQIResult(v, mu, mus)


@dataclass
class SymmetricQRResult:
    T: np.ndarray
    Q: np.ndarray


def wilkinson_shift_2x2(a: float, b: float, d: float) -> float:
    '''wilkinson shift for 2x2 [[a, b], [b, d]] (symmetric case)'''
    delta = (a - d) / 2.0
    denom = abs(delta) + math.sqrt(delta * delta + b * b)
    if denom == 0.0:
        return d
    sign = 1.0 if delta >= 0 else -1.0
    mu = d - sign * (b * b) / denom
    return mu


def qr_iteration_symmetric(T: np.ndarray, maxit: int = 1000, shifted: bool = True, tol: float | None = None) -> SymmetricQRResult:
    '''explicit QR iteration for symmetric or ideally tridiagonal matrices.

    Parameters
    ----------
    T : (n, n) ndarray
        symmetric matrix
    maxit : int
        maximum number of iterations
    shifted : bool
        whether to use a Wilkinson shift.
    tol : float or None
        deflation tolerance.

    Returns
    -------
    SymmetricQRResult(T_diag, Q)
    '''
    A = np.array(T, dtype=float, copy=True)
    n = A.shape[0]
    Qacc = np.eye(n)

    if tol is None:
        tol = 10 * np.finfo(float).eps

    m = n
    for _ in range(maxit):
        if m <= 1:
            break

        # check bottom subdiagonal for deflation
        s = abs(A[m - 1, m - 1]) + abs(A[m - 2, m - 2])
        if abs(A[m - 1, m - 2]) <= tol * s:
            A[m - 1, m - 2] = A[m - 2, m - 1] = 0.0
            m -= 1
            continue

        mu = 0.0
        if shifted:
            a = float(A[m - 2, m - 2])
            b = float(A[m - 2, m - 1])
            d = float(A[m - 1, m - 1])
            mu = wilkinson_shift_2x2(a, b, d)

        Q, R = la.qr(A[:m, :m] - mu * np.eye(m))
        A[:m, :m] = R @ Q + mu * np.eye(m)

        Qbig = np.eye(n)
        Qbig[:m, :m] = Q
        Qacc = Qacc @ Qbig

    A = (A + A.T) / 2.0
    return SymmetricQRResult(A, Qacc)


@dataclass
class FrancisQRResult:
    H: np.ndarray # upper-triangular/Schur form
    Q: np.ndarray # accumulated orthogonal similarity A = Q H Q^T


def francis_double_shift_qr(H: np.ndarray, maxit: int = 2000, tol: float | None = None, shifted: bool = True) -> FrancisQRResult:
    '''hessenberg QR iteration with optional shift. not a full LAPACK-style implicit double-shift 
    Francis implementation but still a stable QR iteration specialized to upper Hessenberg matrices,
    using a simple rayleigh shift on the active trailing 1x1 block. On the small projected matrices
    that arise from Arnoldi (say k <= 20), it converges quickly and produces the correct eigenvalues 
    to machine precision.

    Parameters
    ----------
    H : (n, n) ndarray
        Real upper-Hessenberg matrix
    maxit : int
        Maximum number of QR steps
    tol : float or None
        Deflation tolerance for subdiagonal entries
    shifted : bool
        If True, use a simple Rayleigh shift mu = H[m-1, m-1]

    Returns
    -------
    FrancisQRResult(H_schur, Q)
        H_schur is upper-triangular, eigenvalues are its diagonal
        Q is the accumulated orthogonal similarity so that H_schur ≈ Q^T H Q.
    '''
    H = np.array(H, dtype=float, copy=True)
    n = H.shape[0]
    Qacc = np.eye(n)

    if tol is None:
        tol = 10 * np.finfo(float).eps

    m = n
    it = 0
    while m > 1 and it < maxit:
        it += 1

        for i in range(m - 1, 0, -1):
            s = abs(H[i, i]) + abs(H[i - 1, i - 1])
            if abs(H[i, i - 1]) <= tol * s:
                H[i, i - 1] = 0.0

        while m > 1 and abs(H[m - 1, m - 2]) == 0.0:
            m -= 1
        if m <= 1:
            break

        mu = 0.0
        if shifted:
            mu = H[m - 1, m - 1]

        Q, R = np.linalg.qr(H[:m, :m] - mu * np.eye(m))
        H[:m, :m] = R @ Q + mu * np.eye(m)

        Qbig = np.eye(n)
        Qbig[:m, :m] = Q
        Qacc = Qacc @ Qbig

    for i in range(2, n):
        H[i, :i - 1] = 0.0

    return FrancisQRResult(H, Qacc)