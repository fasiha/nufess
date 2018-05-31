"""
See [Fessler]: https://ieeexplore.ieee.org/document/1166689/:

Fessler, Sutton, "Nonuniform fast Fourier transforms using min-max interpolation", 2003,
IEEE Trans. Signal Processing.

And https://web.eecs.umich.edu/~fessler/code/ for Matlab/MEX.
"""

import numpy as np
import numpy.fft as fft
from scipy.special import jv, iv


def kaiserBesselFourierTransform(u, alpha, m, nneighbors, ndim):
    "Ψ(u) = Psi(u), p. 571 of [Fessler], modified to match their Matlab code"
    z = np.sqrt((np.pi * nneighbors * (u + 0j))**2 - alpha**2)
    nu = ndim / 2 + m
    LambdaZ = jv(nu, z) / z**(nu)
    sol = ((2 * np.pi)**
           (ndim * 0.5)) * ((nneighbors * 0.5)**ndim) * (alpha**m) * LambdaZ / iv(m, alpha)
    return sol.real


def kaiserBessel(k, alpha, m, nneighbors):
    "ψ(κ) = psi(k), p. 570 of [Fessler]"
    mask = np.abs(k) < (0.5 * nneighbors)
    f = np.sqrt(0j + 1 - (k[mask] / nneighbors * 2)**2)
    sol = (f**m) * iv(m, alpha * f) / iv(m, alpha)
    sol2 = np.zeros(k.shape, sol.real.dtype)
    sol2[mask] = sol.real
    return sol2


def setup(xunif, f1nonunif, nneighbors):
    """Evaluate NUFFT on a vector at normalized frequencies with specified neighborhood

    Outputs M-long vector equivalent to:

    `np.dot(np.exp(-1j * 2 * np.pi * f1nonunif[:, np.newaxis] * np.arange(N)), xunif)`

    where `M = len(f1nonunif)` and `N = len(xunif)`.
    """
    nunif = xunif.size
    nupsampled = 2 * nunif  # this is "flexible" in the paper but not really
    etaSpacing = (nunif - 1) / 2
    n = np.arange(nunif)  # 0 ... N-1
    kbftArgs = (n - etaSpacing) / nupsampled
    kbAlpha = 2.34 * nneighbors  # rule of thumb
    kbOrder = 0  # rule of thumb
    prescale = kaiserBesselFourierTransform(kbftArgs, kbAlpha, kbOrder, nneighbors, 1)

    offset = np.floor(f1nonunif * nupsampled - nneighbors / 2)
    kbArgs = -np.arange(1, nneighbors + 0.5) + (f1nonunif * nupsampled - offset)[:, np.newaxis]
    Tr = kaiserBessel(kbArgs, kbAlpha, kbOrder, nneighbors)
    postfilter = np.exp(1j * 2 * np.pi / nupsampled * etaSpacing * kbArgs) * Tr

    Y = fft.fft(xunif / prescale, nupsampled)
    Xnonunif = np.zeros(f1nonunif.shape, Y.dtype)
    for idx in range(len(f1nonunif)):
        o = offset[idx]
        Xnonunif[idx] = np.dot(Y[np.mod(
            np.arange(o + 1, 1 + o + nneighbors, dtype=int), nupsampled)], np.conj(postfilter[idx]))
    return Xnonunif


if __name__ == '__main__':
    N = 32
    xunif = np.arange(N, dtype=float) + 1j * np.random.randn(N)
    fractionalbins = np.array([0, 1, 2.0, 3]) / 4.4  # fractional FFT index
    f1nonunif = fractionalbins / N  # FFT index / N
    nneighbors = 9
    Xn = setup(xunif, f1nonunif, nneighbors)
    gold = np.dot(np.exp(-1j * 2 * np.pi * f1nonunif[:, np.newaxis] * np.arange(N)), xunif)
    assert np.allclose(Xn, gold)
    print('1D unit test passed')