import numpy as np
from scipy.special import jv, iv


def kaiserBesselFourierTransform(u, alpha, m, nneighbors, ndim):
    "Ψ(u) = Psi(u), p. 571 of [Fessler]"
    z = np.sqrt((np.pi * nneighbors * (u + 0j))**2 - alpha**2)
    nu = ndim / 2 + m
    LambdaZ = (z * 0.5)**(-nu) * jv(nu, z)
    sol = (0.5**m) * (np.pi**(ndim * 0.5)) * ((nneighbors * 0.5)**ndim) * (
        alpha**m) * LambdaZ / iv(m, alpha)
    return sol.real


def setup(xunif, fnonunif, nneighbors, ntable):
    N = xunif.size
    K = 2 * N  # this is "flexible" in the paper but not really
    eta0 = (N - 1) / 2
    n = np.arange(N)  # 0 ... N-1
    kbftArgs = (n - eta0) / K
    alpha = 2.34 * nneighbors  # rule of thumb
    m = 0  # rule of thumb
    scaling = kaiserBesselFourierTransform(kbftArgs, alpha, m, nneighbors, 1)
    return scaling


if __name__ == '__main__':
    s = setup(np.random.randn(32), [], 4, 2**14)
    print(s)
