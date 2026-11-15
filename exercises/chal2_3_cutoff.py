# -*- coding: utf-8 -*-
"""
Challenge 2.3 -- Cusps, cell size and the cut-off energy               [Challenge]

Outline only; see the description for the questions to answer.
"""

import matplotlib.pyplot as plt
import numpy as np

import pwlib


def hydrogenic(r, M, R, Z=1.0):
    """Normalized hydrogen-like 1s orbital sqrt(Z^3/pi) exp(-Z|r-R|)."""
    d = np.linalg.norm(pwlib.minimum_image(r, R, M), axis=-1)
    return np.sqrt(Z**3 / np.pi) * np.exp(-Z * d)


def gaussian(r, M, R):
    """The normalized Gaussian wave function g(r) of the primer."""
    return np.sqrt(pwlib.gaussian_density(r, M, R, alpha=2.0))


def metrics(orbital, a0, N, **kw):
    """TODO: return (norm in real space, sum |c|^2, kinetic energy)."""
    raise NotImplementedError


def fraction_outside_cube(a0, Z=1.0, n=120):
    """
    TODO: 1 - integral of |psi_1s|^2 over the cube. Suggestion: Gauss-Legendre
    quadrature (np.polynomial.legendre.leggauss) over one octant [0, a0/2]^3,
    times eight.
    """
    raise NotImplementedError


def main():
    a0 = 10.0
    Ns = np.array([8, 16, 32, 64, 128])
    # (a) relative errors of the three metrics for the 1s orbital versus N;
    #     check your N = 128 values against the primer: norm 0.99919444,
    #     E_kin 0.49990246
    # (b) the fraction of the orbital outside the cell
    # (c) the same for the Gaussian
    # (d) spherical cut-off: E_kin(E_cut) and N_pw(E_cut) for Z = 1, 2, 3,
    #     using the analytical coefficients
    #       c(G) = 8 sqrt(pi Z^5) / (Z^2 + |G|^2)^2 / sqrt(Omega)
    plt.show()


if __name__ == '__main__':
    main()
