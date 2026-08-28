import numpy as np

def arnoldi(A, b, k_max, tol=1e-14):
    '''arnoldi iteration, modified Gram-Schmidt form

    Parameters
    ----------
    A : (m, m) ndarray
        matrix real or complex
    b : (m,) ndarray
        Starting vector, nonzero
    k_max : int
        maximum number of arnoldi steps
    tol : float
        breakdown tolerance

    Returns
    -------
    Q : (m, k_eff+1) ndarray
        qrthonormal basis vectors [q_1, ..., q_{k_eff+1}]
    Htilde : (k_eff+1, k_eff) ndarray
        upper hessenberg matrix such that A Q_k = Q_{k+1} Htilde_k
    '''
    
    m = A.shape[0]

    Htilde = np.zeros((k_max + 1, k_max), dtype = A.dtype)
    Q = np.zeros((m,k_max + 1), dtype = A.dtype)
    
    q = b/np.linalg.norm(b)
    Q[:,0] = q
    
    k_eff = k_max

    for k in range(k_max):
        v = A @ Q[:,k]
        for j in range(0,k + 1):
            Htilde[j,k] = Q[:,j].conj().T @ v
            v = v - Htilde[j,k]*Q[:,j]

        Htilde[k+1,k] = np.linalg.norm(v)

        if Htilde[k+1,k] < tol:
            k_eff = k
            break
        else:
            q = v/Htilde[k+1,k]
            Q[:, k + 1] = q
    
    Q_eff = Q[:, : (k_eff+1)]
    Htilde = Htilde[: (k_eff+1), : k_eff]

    return Q_eff, Htilde


def lanczos(A, b, k_max, tol=1e-14):
    '''lanczos iteration for real symmetric matrices

    Parameters
    ----------
    A : (m, m) ndarray
        real symmetric matrix
    b : (m,) ndarray
        starting vector, nonzero
    k_max : int
        maximum number of lanczos steps
    tol : float
        breakdown tolerance

    Returns
    -------
    Q : (m, k) ndarray
        orthonormal lanczos vectors [q_1, ..., q_k]
    T : (k, k) ndarray
        symmetric tridiagonal matrix T_k = Q^T A Q
    alpha : (k,) ndarray
        diagonal entries of T
    beta : (k+1,) ndarray
        off diagonal entries beta[1..k-1] used where beta[0]=0, beta[k] is last norm
    '''
    
    m = A.shape[0]

    Q = np.empty((m, 0))
    alpha = np.zeros(k_max+1)
    beta  = np.zeros(k_max+1)

    beta[0] = 0
    qlast = np.zeros(m)
    q = b/np.linalg.norm(b)

    k = 0
    for k in range(1, k_max+1):

        v = A @ q

        alpha[k] = np.vdot(q, v)

        v = v - beta[k-1] * qlast - alpha[k] * q

        beta[k] = np.linalg.norm(v)

        Q = np.hstack((Q,q.reshape(-1,1)))

        if beta[k] < tol:
            break
        else:
            qlast = q.copy()
            q = v/beta[k]


    Q = Q[:, :k]
    alpha = alpha[1:k+1]
    beta = beta[1:k]

    T = np.diag(alpha) + np.diag(beta, k=-1) + np.diag(beta, k=1)


    return Q, T, np.array(alpha), np.array(beta)

import numpy as np

def lanczos_reorthogonalization(A, b, k_max, tol=1e-14):
    '''lanczos iteration for real symmetric matrices, with full reorthogonalization

    Parameters
    ----------
    A : (m, m) ndarray
        real symmetric matrix
    b : (m,) ndarray
        starting vector, nonzero
    k_max : int
        maximum number of lanczos steps
    tol : float
        breakdown tolerance

    Returns
    -------
    Q : (m, k) ndarray
        orthonormal lanczos vectors [q_1, ..., q_k]
    T : (k, k) ndarray
        symmetric tridiagonal matrix T_k = Q^T A Q
    alpha : (k,) ndarray
        diagonal entries of T
    beta : (k-1,) ndarray
        off diagonal entries of T
    '''

    m = A.shape[0]

    Q = np.empty((m, 0))
    alpha = np.zeros(k_max + 1)
    beta = np.zeros(k_max + 1)

    beta[0] = 0
    qlast = np.zeros(m)
    q = b / np.linalg.norm(b)

    k = 0
    for k in range(1, k_max + 1):

        v = A @ q

        alpha[k] = np.vdot(q, v)

        v = v - beta[k - 1] * qlast - alpha[k] * q

        if Q.shape[1] > 0:
            v -= Q @ (Q.T @ v)
            v -= Q @ (Q.T @ v)

        beta[k] = np.linalg.norm(v)

        Q = np.hstack((Q, q.reshape(-1, 1)))

        if beta[k] < tol:
            break
        else:
            qlast = q.copy()
            q = v / beta[k]

    Q = Q[:, :k]
    alpha = alpha[1:k + 1]
    beta = beta[1:k]

    T = np.diag(alpha) + np.diag(beta, k=-1) + np.diag(beta, k=1)

    return Q, T, np.array(alpha), np.array(beta)


def conjugate_gradient(A, b, x0=None, tol=1e-8, maxiter=None):
    '''conjugate gradient method for SPD matrices

    Parameters
    ----------
    A : (m, m) ndarray
        symmetric positive definite matrix
    b : (m,) ndarray
        rhs 
    x0 : (m,) ndarray or None
        initial guess. If None, use zero
    tol : float
        stopping tolerance on residual norm
    maxiter : int or None
        maximum number of iterations. If None, use m.

    Returns
    -------
    x : (m,) ndarray
        Approximate solution.
    res_norms : list of float
        Residual norms at each iteration (including initial one).
    xs : list of ndarray
        Iterates x_k (for inspection / plotting).
    '''
    
    m = A.shape[0]

    if x0 is None:
        x0 = np.zeros(m)
    if maxiter is None:
        maxiter = m
    
    res_norms = np.zeros(maxiter + 1)
    xs = np.zeros((m,maxiter + 1))
    r = np.zeros((m,maxiter + 1))
    p = np.zeros((m,maxiter + 1))
    alpha = np.zeros(maxiter + 1)
    beta = np.zeros(maxiter + 1)

    alpha[0] = 0
    beta[0] = 0
    xs[:, 0] = x0

    r[:, 0] = b - A @ x0
    p[:, 0] = r[:, 0].copy()

    res_norms[0] = np.linalg.norm(r[:, 0])

    n = 0
    for n in range(1, maxiter + 1):
        alpha[n] = (np.dot(r[:, n-1], r[:, n-1])) / (p[:, n-1].T @ A @ p[:, n-1])
        xs[:, n] = xs[:, n-1] + alpha[n] * p[:, n-1]
        r[:, n] = r[:, n-1] - alpha[n] * A @ p[:, n-1]
        beta[n] = (np.dot(r[:, n], r[:, n])) / (np.dot(r[:, n-1], r[:, n-1]))
        p[:, n] = r[:, n] + beta[n] * p[:, n-1]

        res_norms[n] = np.linalg.norm(r[:,n])

        if res_norms[n] < tol:
            break

    x = xs[:, n]

    print(f'max iterations to achieve tolerance or reach maxiter {n}')

    return x, res_norms, xs


def gmres(A, b, x0=None, k_max=None, tol=1e-8):
    '''unrestarted GMRES for general systems

    Parameters
    ----------
    A : (m, m) ndarray
        matrix, not necessarily symmetric
    b : (m,) ndarray
        rhs
    x0 : (m,) ndarray or None
        initial guess. If None, use zero
    k_max : int or None
        maximum Krylov dimension. If None, use m
    tol : float
        stopping tolerance on residual norm

    Returns
    -------
    x : (m,) ndarray
        approximate solution
    res_norms : list of float
        residual norms at each GMRES step
    '''
    m = A.shape[0]

    if x0 is None:
        x0 = np.zeros(m)
    if k_max is None:
        k_max = m
    
    r0 = b - A @ x0
    beta = np.linalg.norm(r0)

    if beta == 0:
        return x0, [0.0]

    Htilde = np.zeros((k_max + 1, k_max), dtype=A.dtype)
    Q = np.zeros((m, k_max + 1), dtype=A.dtype)

    cs = np.zeros(k_max, dtype=A.dtype)
    sn = np.zeros(k_max, dtype=A.dtype)

    g = np.zeros(k_max + 1, dtype=A.dtype)
    g[0] = beta

    res_norms = [beta]
    Q[:, 0] = r0 / beta

    def apply_givens(c, s, v0, v1):
        v0_new =  c * v0 + s * v1
        v1_new = -np.conj(s) * v0 + np.conj(c) * v1
        return v0_new, v1_new

    k_final = 0

    for k in range(k_max):
        v = A @ Q[:, k]

        for j in range(0, k + 1):
            Htilde[j, k] = Q[:, j].conj().T @ v
            v = v - Htilde[j, k] * Q[:, j]

        Htilde[k + 1, k] = np.linalg.norm(v)

        if Htilde[k + 1, k] < tol:
            k_final = k
            break
        
        Q[:, k + 1] = v / Htilde[k + 1, k]

        for i in range(k):
            Htilde[i, k], Htilde[i + 1, k] = apply_givens(
                cs[i], sn[i], Htilde[i, k], Htilde[i + 1, k]
            )

        a = Htilde[k, k]
        b_elem = Htilde[k + 1, k]

        r = np.sqrt((a * a.conjugate()).real + (b_elem * b_elem.conjugate()).real)
        if r == 0:
            c = 1.0
            s = 0.0
        else:
            c = a / r
            s = b_elem / r

        cs[k] = c
        sn[k] = s

        Htilde[k, k], Htilde[k + 1, k] = apply_givens(
            c, s, Htilde[k, k], Htilde[k + 1, k]
        )

        g[k], g[k + 1] = apply_givens(c, s, g[k], g[k + 1])

        res_norm = abs(g[k + 1])
        res_norms.append(res_norm)

        if res_norm < tol:
            k_final = k
            break

        k_final = k

    R = Htilde[:k_final + 1, :k_final + 1]
    rhs = g[:k_final + 1]

    y = np.zeros(k_final + 1, dtype=A.dtype)
    for i in range(k_final, -1, -1):
        y[i] = (rhs[i] - np.dot(R[i, i + 1:], y[i + 1:])) / R[i, i]

    x_n = x0 + Q[:, :k_final + 1] @ y

    return x_n, res_norms