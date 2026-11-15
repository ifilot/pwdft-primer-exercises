# -*- coding: utf-8 -*-
"""
Challenge 2.3 -- Cusps, cell size and the cut-off energy             [SOLUTION]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

OUTPUT = Path(__file__).resolve().parent / 'output'


def hydrogenic(r, M, R, Z=1.0):
    """Normalized hydrogen-like 1s orbital sqrt(Z^3/pi) exp(-Z|r-R|)."""
    d = np.linalg.norm(pwlib.minimum_image(r, R, M), axis=-1)
    return np.sqrt(Z**3 / np.pi) * np.exp(-Z * d)


def gaussian(r, M, R):
    """The normalized Gaussian wave function g(r) of the primer."""
    return np.sqrt(pwlib.gaussian_density(r, M, R, alpha=2.0))


def metrics(orbital, a0, N, **kw):
    M = pwlib.cubic_cell(a0)
    r = pwlib.realspace_grid(M, N)
    _, G2 = pwlib.gvectors(M, N)
    psi = orbital(r, M, np.full(3, a0 / 2), **kw)
    c = pwlib.fft_pw(psi, M)
    return pwlib.integrate(psi**2, M), np.sum(np.abs(c)**2), pwlib.kinetic_energy(c, G2)


def fraction_outside_cube(a0, Z=1.0, n=120):
    """1 - integral of |psi_1s|^2 over the cube, by Gauss-Legendre quadrature
    over one octant (the cusp sits in a corner of the octant)."""
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.25 * a0 * (x + 1.0)              # map [-1, 1] -> [0, a0/2]
    w = 0.25 * a0 * w
    X, Y, Zc = np.meshgrid(x, x, x, indexing='ij')
    W = w[:, None, None] * w[None, :, None] * w[None, None, :]
    d = np.sqrt(X**2 + Y**2 + Zc**2)
    return 1.0 - 8.0 * np.sum(W * Z**3 / np.pi * np.exp(-2.0 * Z * d))


def main():
    a0 = 10.0
    Ns = np.array([8, 16, 32, 64, 128])

    # (a) + (c): reproduce the convergence figure, and compare with a Gaussian
    h = np.array([metrics(hydrogenic, a0, N) for N in Ns])
    g = np.array([metrics(gaussian, a0, N) for N in Ns])
    outside = fraction_outside_cube(a0)
    print('fraction of the 1s orbital outside the cell: %.4e' % outside)
    pwlib.check('E_kin(1s), N = 128 (primer, Fig. 4)', h[-1, 2], 0.49990246, 1e-8)
    pwlib.check('norm(1s), N = 128 (primer, Fig. 4)', h[-1, 0], 0.99919444, 1e-8)
    pwlib.check('plateau of the norm = 1 - fraction outside', 1 - outside,
                0.99918943, 1e-6)
    print(' N   rel.err. norm(1s)  rel.err. Ekin(1s)  rel.err. Ekin(Gaussian)')
    for N, hh, gg in zip(Ns, h, g):
        print('%3i  %+.3e         %+.3e         %+.3e'
              % (N, hh[0] - 1, (hh[2] - 0.5) / 0.5, (gg[2] - 1.5) / 1.5))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(Ns, np.abs(h[:, 0] - 1), 'o-', label='norm, 1s')
    ax[0].loglog(Ns, np.abs(h[:, 2] - 0.5) / 0.5, 's-', label=r'$E_\mathrm{kin}$, 1s')
    ax[0].loglog(Ns, np.abs(g[:, 2] - 1.5) / 1.5 + 1e-16, '^-',
                 label=r'$E_\mathrm{kin}$, Gaussian')
    ax[0].axhline(outside, ls='--', color='k', lw=0.8, label='fraction outside cell')
    ax[0].set_xlabel('N per direction')
    ax[0].set_ylabel('relative error')
    ax[0].legend(fontsize=8)

    # (d) spherical cut-off; to separate the cut-off from grid and cell-size
    # effects, use the analytical coefficients of the hydrogen-like orbital,
    #   c(G) = 8 sqrt(pi Z^5) / (Z^2 + |G|^2)^2 / sqrt(Omega)
    N = 192
    M = pwlib.cubic_cell(a0)
    _, G2 = pwlib.gvectors(M, N)
    ecuts = np.geomspace(5, 0.5 * G2.max() / 3, 40)   # spheres inside the grid
    print('\n Z   E_cut for 1% error in E_kin   E_cut / Z^2')
    e1s = []
    for Z in (1.0, 2.0, 3.0):
        c = 8.0 * np.sqrt(np.pi * Z**5) / (Z**2 + G2)**2 / np.sqrt(a0**3)
        w = 0.5 * c**2 * G2
        ekin = np.array([np.sum(w[0.5 * G2 <= e]) for e in ecuts])
        # measure the error relative to the complete sum, such that only the
        # effect of the cut-off remains (cf. the cell-size error in part (b))
        etot = np.sum(w)
        print('    complete sum E_kin = %.6f (Z^2/2 = %.6f)' % (etot, 0.5 * Z**2))
        err = (etot - ekin) / etot
        e1 = np.exp(np.interp(np.log(0.01), np.log(err[::-1]),
                              np.log(ecuts[::-1])))
        e1s.append(e1 / Z**2)
        print('%2i   %8.1f                     %6.2f' % (Z, e1, e1 / Z**2))
        ax[1].loglog(ecuts, err, 'o-', ms=3, label='Z = %i' % Z)
        if Z == 1.0:
            npw = np.array([np.sum(0.5 * G2 <= e) for e in ecuts])
            slope = np.polyfit(np.log(ecuts[5:]), np.log(npw[5:]), 1)[0]
            pwlib.check('N_pw grows as E_cut^(3/2)', slope, 1.5, 0.02)
    pwlib.check('E_cut(Z=3) / E_cut(Z=1) = 9', e1s[2] / e1s[0] * 9, 9.0, 0.5)
    ax[1].set_xlabel(r'$E_\mathrm{cut}$ [Ht]')
    ax[1].set_ylabel(r'relative error $E_\mathrm{kin}$')
    ax[1].legend()
    for a in ax:
        a.grid(ls='--', alpha=0.5, which='both')
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'chal2_3_cutoff.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
