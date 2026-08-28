# Krylov Subspace Algorithms for MNIST Data Analysis

A from scratch numpy implementation of core Krylov subspace methods. We implement Arnoldi, Lanczos, Conjugate Gradient, and Generalized Minimal Residual, and validate/experiment with them on synthetic random matrices with known dense solutions. The algorithms are then applied to the real world problem of computing leading eigenvectors of large scale covariance matrices using the MNIST handwritten digit dataset.

### Key findings

- Arnoldi, Lanczos, CG, and GMRES implemended from scratch generally match theoretical convergence bounds and compare well to dense solvers on random matrices.
- Lanczos applied to a $784 \times 784$ MNIST covariance matrix recovers the leading eigenvector to near machine precision (relative error $\sim 10^{-15}$) without needing a dense eigendecomposition.
- Lanczos experiences a known failure mode where loss of orthogonality leads to duplicate 'ghost' eigenvalues which is identified and diagnosed by comparing eigendigits across Krylov dimensions.

## Motivation

Solving large linear systems or eigenvalue problems is a common computational task that can serve as a bottleneck on large data, the most common being PCA with large covariance matrices. Classical dense linear system (like LU or Gaussian elimination) and eigenvalue solvers (like QR) are computationally expensive, usually scaling around $O(n^3)$. Krylov subspace algorithms provide a computational efficient approximation alternative. We aim to implement these methods from first principles and validate their behavior in accordance with the mathematical theory on by synthetic and real world examples.

## Results

We first experiment on the four core algorithms using synthetic random matrices, then apply the validated solvers to MNIST.

- **Arnoldi converges extremes first, interior eigenvalues lag.** On a random $40\times40$ non-symmetric matrix ($k=5,10,15,20$), Ritz values converge to extreme and well-separated eigenvalues first, while interior eigenvalues converge more slowly and less predictably. The by-hand Francis-QR solver matches numpy's dense solver on the same projected matrix with max Ritz-value error below $2\times10^{-13}$ at $k=5$, validating the custom eigensolver.

- **Lanczos shows the same extreme-first pattern on SPD matrices.** Comparing well-conditioned ($\kappa\approx4.7$) and ill-conditioned ($\kappa\approx109$) $30\times30$ examples, extreme eigenvalues converge within the smallest tested Krylov space ($k=7$) in both cases, while interior eigenvalues converge more slowly, especially when clustered.

- **CG iteration counts track the theoretical bound, except when finite precision breaks it.** Across three condition numbers ($\kappa\approx1.9$, $4.7$, $6659$), iteration counts came out to (16, 26, 74). Against the theoretical worst-case bound $n=O(\sqrt{\kappa}\log(1/\epsilon))$, the well- and moderately-conditioned cases fell below the bound as expected. The ill-conditioned case exceeded the matrix dimension (50), which is only possible due to finite-precision loss of Krylov-subspace independence — a reasonable result given the conditioning of the matrix.

- **GMRES residuals decay log-linearly until conditioning breaks convergence.** Across a range of $k_{\max}$ on non-symmetric and SPD systems, $\log\|r_k\|$ vs. iteration count fits a linear regression with $R^2 \geq 0.9958$ across all tested $k_{\max}$ (slope $\approx-1.4$) in the well-conditioned case, confirming near log-linear convergence. The ill-conditioned case did not reach tolerance even at the largest tested $k_{\max}=50$.

- **Lanczos on MNIST recovers interpretable 'eigendigits,' but only the top mode is trustworthy without reorthogonalization.** Centering a 30,000-image MNIST subset to form the $784\times784$ pixel covariance matrix and running Lanczos ($k=20$) with Ritz vectors lifted back to pixel space produces recognizable eigendigits. Sweeping sample size ($n=10\text{k}$–$60\text{k}$) leaves the eigendigits qualitatively stable, with only modestly less visual noise at larger $n$. Sweeping Krylov dimension ($k=20$ to $200$) tells a different story: eigendigits are sharp and distinct at $k=20$, but as $k$ increases, Ritz values for PC2–PC6 progressively merge onto PC1's eigenvalue (all six coincide by $k=100$, each with residual $\sim10^{-9}$), and the eigendigits collapse into six near-identical copies. This is a textbook Lanczos ghost-eigenvalue effect caused by loss of orthogonality, expected given the original implementation has no reorthogonalization step; adding reorthogonalization recovers clean principal components that mirror the dense solver with small residuals.

- **Benchmarking against a dense solver confirms only the top eigenpair is well-converged at this $k$.** Lanczos ($k=20$, $n=30\text{k}$) matches the dense eigendecomposition's leading eigenvalue to $\sim10^{-15}$ relative error, but PCs 2–6 show 39%–130% relative error.

- **Per-digit analysis surfaces structurally meaningful, class-specific components.** Repeating the analysis per digit class (0–9), leading within-class PCs capture stroke start/end position variability and each digit's core structural strokes (e.g. the top bar and slanted leg for '7'), while later PCs are visibly noisier, reflecting finer-grained variation like stroke thickness.

## Limitations and future work

- Full reorthogonalization is used to address the loss of orthogonality issue found in the MNIST analysis which is effective but would serve as a significant bottleneck at the scale of modern deep learning or large data problems. Reorthogonalization as implemented here uses Gram-Schmidt at every step which is extremely costly in both runtime and memory overhead. The natural choice in situations where this overhead is significant would be to use selective reorthogonalization or implicit restarting.

- GMRES is unrestarted so its cost and memory increase with iteration count at the scale of $O(k)$ stored vectors and $O(k^2)$ work per step, again limiting the real applicability to problems at scale.

- The Fransis double shift QR implementation does not use a full LAPACK style implicit double-shift Francis implementation, but it is still a stable QR iteration specialized to upper Hessenberg matrices that uses a simple rayleigh shift on the active trailing 1x1 block.

- While the algorithms implemented here have significant advantages for large data, they also suffer from a number of issues, many documented here, that push them out of modern machine learning pipelines. There exist many optimized versions of these core algorithms for specific taks, but in most current workflows even those are replaced by modern approaches like randomized SVDs, stochastic gradient methods, and neural operators.

## Repository structure

```
krylov-mnist/
|-- src/
|   |-- eigensolvers.py # from scratch dense eigenvalue solvers
|   |-- krylov_algorithms.py # from scratch core krylov algorithms
|   |-- utils.py # linear algebra utils
|-- krylov_methods.ipynb # validation of krylov algorithms
|-- MNIST_analysis.ipynb # MNIST application
```

## Reproducing this project

```bash
# 1. clone and install
git clone <repo-url>
pip install -r requirements.txt

# 2. place noaa data in data/

# 3. run krylov methods and MNIST notebooks
```


