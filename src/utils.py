import math
import numpy as np
from typing import Optional

def make_spd_matrix(n: int, seed: int = 0, shift: Optional[float] = None) -> np.ndarray:
    '''make a random symmetric positive definite (SPD) matrix

    Parameters
    ----------
    n : int
        size of the matrix
    seed : int, optional
        random seed, by default 0
    shift : Optional[float], optional
        shift to apply to improve conditioning, by default None

    Returns
    -------
    np.ndarray
        spd matrix
    '''
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M.T @ M
    # shift the diagonal to improve conditioning
    if shift is None: shift = n
    A += shift * np.eye(n)
    return A


def make_nonsymmetric_matrix(n: int, seed: int = 0) -> np.ndarray:
    '''make a random nonsymmetric matrix

    Parameters
    ----------
    n : int
        size of the matrix
    seed : int, optional
        random seed, by default 0

    Returns
    -------
    np.ndarray
        nonsymmetric matrix
    '''
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A


def relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    '''compute the relative residual norm ||b - Ax|| / ||b||

    Parameters
    ----------
    A : np.ndarray
        coefficient matrix
    x : np.ndarray
        candidate solution
    b : np.ndarray
        right-hand side

    Returns
    -------
    float
        relative residual norm
    '''
    r = b - A @ x
    return float(np.linalg.norm(r) / np.linalg.norm(b))


def schur_eigs_real(H_schur: np.ndarray, tol: float = 1e-12) -> np.typing.NDArray[np.complexfloating]:
    '''extract eigenvalues from a real quasi-upper-triangular Schur form

    Given a real quasi-upper-triangular Schur form H (1x1 and 2x2 blocks),
    return the eigenvalues as a 1D complex array.

    Parameters
    ----------
    H_schur : np.ndarray
        real quasi-upper-triangular Schur form
    tol : float, optional
        tolerance for detecting 2x2 blocks, by default 1e-12

    Returns
    -------
    np.ndarray
        eigenvalues as a 1D complex array
    '''
    H = np.array(H_schur, dtype=float, copy=False)
    n = H.shape[0]
    eigs = []
    i = 0
    while i < n:
        if i == n - 1 or abs(H[i+1, i]) < tol:
            eigs.append(H[i, i])
            i += 1
        else:
            # block [[a, b], [c, d]]
            a, b = H[i, i],   H[i, i+1]
            c, d = H[i+1, i], H[i+1, i+1]
            trace = a + d
            det   = a*d - b*c
            disc  = trace*trace - 4.0*det
            if disc >= 0:
                root = math.sqrt(disc)
            else:
                root = complex(0.0, math.sqrt(-disc))
            eig1 = 0.5*(trace + root)
            eig2 = 0.5*(trace - root)
            eigs.extend([eig1, eig2])
            i += 2
    return np.array(eigs, dtype=complex)