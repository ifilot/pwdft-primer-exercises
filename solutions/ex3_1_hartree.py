# -*- coding: utf-8 -*-
"""
Exercise 3.1 -- The Hartree potential of a Gaussian charge cloud  [SOLUTION]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

import pwlib

OUTPUT = Path(__file__).resolve().parent / 'output'
XI = 2.837297       # Madelung constant, simple cubic, neutralizing background


def erep_periodic(a0, r2=0.75):
    """Expected repulsion energy of the periodic system (primer, Section 3)."""
    return 2.0 / np.sqrt(np.pi) - XI / a0 + 4.0 * np.pi * r2 / (3.0 * a0**3)


def main():
    a0 = 10.0
    M = pwlib.cubic_cell(a0)
    centre = np.array([5.0, 5.0, 5.0])

    # (a) repulsion energy for increasing N
    print(' N    E_rep            E_rep - expected')
    for N in (16, 24, 32, 48):
        r = pwlib.realspace_grid(M, N)
        _, G2 = pwlib.gvectors(M, N)
        rho = pwlib.gaussian_density(r, M, centre, alpha=2.0)
        vh = pwlib.hartree_potential(rho, G2)
        erep = pwlib.integrate(vh * rho, M)
        print('%3i  %.10f  %+.2e' % (N, erep, erep - erep_periodic(a0)))

    N = 48
    r = pwlib.realspace_grid(M, N)
    _, G2 = pwlib.gvectors(M, N)
    rho = pwlib.gaussian_density(r, M, centre, alpha=2.0)
    vh = pwlib.hartree_potential(rho, G2)
    erep = pwlib.integrate(vh * rho, M)
    pwlib.check('E_rep vs the periodic-system formula (Section 3), N = 48', erep,
                erep_periodic(a0), 1e-7)
    pwlib.check('E_H = E_rep / 2 vs Table in Section 4 (L = 10)', 0.5 * erep,
                0.4239, 5e-5)
    pwlib.check('E_rep - E_rep^iso (Madelung shift)', erep - 2 / np.sqrt(np.pi),
                -XI / a0 + np.pi / a0**3, 1e-6)

    # (b) the potential itself
    pwlib.check('cell average of v_H vanishes', np.mean(vh), 0.0, 1e-12)
    x = r[:, N // 2, N // 2, 0]
    d = np.abs(x - 5.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        v_iso = np.where(d > 0, erf(np.sqrt(2.0) * d) / d, 2 * np.sqrt(2 / np.pi))
    v_line = vh[:, N // 2, N // 2]
    shift = v_line[N // 2] - v_iso[N // 2]
    print('v_H - v_iso at the centre of the cloud: %+.4f Ht' % shift)
    print('(compare -xi/a0 + 2 pi <r^2> / (3 a0^3) = %+.4f Ht)'
          % (-XI / a0 + 2 * np.pi * 0.75 / (3 * a0**3)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, v_line, 'o-', label=r'$\nu_\mathrm{H}$ (periodic, plane waves)')
    ax.plot(x, v_iso, '--', label=r'$\mathrm{erf}(\sqrt{2}r)/r$ (isolated)')
    ax.plot(x, v_iso + shift, ':', label='isolated, shifted')
    ax.set_xlabel('x [bohr]  (y = z = 5)')
    ax.set_ylabel('potential [Ht]')
    ax.legend()
    ax.grid(ls='--', alpha=0.5)
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'ex3_1_hartree.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
