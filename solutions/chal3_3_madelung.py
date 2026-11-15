# -*- coding: utf-8 -*-
"""
Challenge 3.3 -- Measuring the Madelung constant                    [SOLUTION]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

OUTPUT = Path(__file__).resolve().parent / 'output'
XI = 2.837297
SPACING = 10.0 / 32          # fixed grid spacing (bohr), converged for alpha >= 1
INCLUDE_1S = False           # optional part (d): needs ~4 GB of memory and minutes


def electrostatics(a0, alpha, density='gaussian'):
    """E_rep and E_nuc of one normalized cloud centred on a proton."""
    N = int(round(a0 / SPACING)) if density == 'gaussian' else int(round(a0 * 12.8))
    M = pwlib.cubic_cell(a0)
    r = pwlib.realspace_grid(M, N)
    G, G2 = pwlib.gvectors(M, N)
    R = np.full(3, a0 / 2)
    if density == 'gaussian':
        rho = pwlib.gaussian_density(r, M, R, alpha)
    else:                                   # hydrogen 1s density
        d = np.linalg.norm(pwlib.minimum_image(r, R, M), axis=-1)
        rho = np.exp(-2.0 * d) / np.pi
    vext = pwlib.external_potential(G, G2, M, [R], [1.0])
    vh = pwlib.hartree_potential(rho, G2)
    return pwlib.integrate(vh * rho, M), pwlib.integrate(vext * rho, M)


def fit(a0s, dE):
    """Least-squares fit dE(a0) = p / a0 + q / a0^3; returns (p, q)."""
    A = np.column_stack([1.0 / a0s, 1.0 / a0s**3])
    return np.linalg.lstsq(A, dE, rcond=None)[0]


def main():
    a0s = np.array([10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0])
    fig, ax = plt.subplots(figsize=(6, 4))

    for alpha in (2.0, 1.0):
        r2 = 1.5 / alpha                    # <r^2> of the normalized Gaussian
        erep_iso = 2.0 * np.sqrt(alpha / (2.0 * np.pi))
        enuc_iso = -2.0 * np.sqrt(alpha / np.pi)
        res = np.array([electrostatics(a, alpha) for a in a0s])
        drep = res[:, 0] - erep_iso
        dnuc = res[:, 1] - enuc_iso

        p, q = fit(a0s, drep)
        print('alpha = %.1f, E_rep: xi = %.6f, q = %.4f (expected %.4f)'
              % (alpha, -p, q, 4 * np.pi * r2 / 3))
        pwlib.check('xi from E_rep (alpha = %.0f)' % alpha, -p, XI, 1e-5)
        pwlib.check('spread term from E_rep (alpha = %.0f)' % alpha, q,
                    4 * np.pi * r2 / 3, 1e-3)
        p, q = fit(a0s, dnuc)
        pwlib.check('xi from E_nuc (alpha = %.0f)' % alpha, p, XI, 1e-5)
        pwlib.check('spread term from E_nuc (alpha = %.0f)' % alpha, q,
                    -2 * np.pi * r2 / 3, 1e-3)

        ax.loglog(a0s, np.abs(drep), 'o', label=r'$|\Delta E_\mathrm{rep}|$, $\alpha=%g$' % alpha)
        ax.loglog(a0s, np.abs(dnuc), 's', mfc='none',
                  label=r'$|\Delta E_\mathrm{nuc}|$, $\alpha=%g$' % alpha)
        ax.loglog(a0s, np.abs(drep + dnuc), '^', label=r'$|\Delta E_\mathrm{sum}|$, $\alpha=%g$' % alpha)

    ax.loglog(a0s, XI / a0s, 'k--', lw=0.8, label=r'$\xi/a_0$')
    ax.set_xlabel(r'$a_0$ [bohr]')
    ax.set_ylabel('deviation from isolated value [Ht]')
    ax.legend(fontsize=7)
    ax.grid(ls='--', alpha=0.5, which='both')
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'chal3_3_madelung.png', dpi=150)

    if INCLUDE_1S:
        # (d) the hydrogen 1s density: cusp requires a much finer grid
        a1 = np.array([10.0, 15.0, 20.0])
        res = np.array([electrostatics(a, None, density='1s') for a in a1])
        p, q = fit(a1, res[:, 1] + 1.0)
        print('1s: xi from E_nuc = %.5f, spread = %.3f (expected %.3f)'
              % (p, q, -2 * np.pi * 3 / 3))
    plt.show()


if __name__ == '__main__':
    main()
