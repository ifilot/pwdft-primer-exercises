# -*- coding: utf-8 -*-
"""
Exercise 7.1 -- The local part of a pseudopotential               [SOLUTION]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

import pwlib

OUTPUT = Path(__file__).resolve().parent / 'output'
XI = 2.837297


def main():
    p = pwlib.GTH_O
    Z, rloc, C = p['Zion'], p['rloc'], p['C']

    # (a) the G = 0 constant and the value at the nucleus
    alpha = pwlib.gth_local_g(np.array([0.0]), Z, rloc, C)[0]
    pwlib.check('alpha (primer, appendix)', alpha, 0.134, 5e-4)
    # nu_loc(0) from YOUR reciprocal-space expression: for a spherical
    # function, nu(0) = (2 pi)^-3 int nu(G) d^3G = 1/(2 pi^2) int nu(G) G^2 dG
    v0 = quad(lambda g: pwlib.gth_local_g(np.array([g * g]), Z, rloc, C)[0] * g * g
              if g > 0 else 0.0, 0.0, 60.0, limit=400)[0] / (2 * np.pi**2)
    pwlib.check('nu_loc(0) from nu_loc(G) (primer: about -36 Ht)', v0, -36.24, 0.01)
    pwlib.check('... equals the real-space value', v0,
                pwlib.gth_local_r(0.0, Z, rloc, C), 1e-6)

    # (b) reciprocal space: compare with the Coulomb tails
    G2 = np.geomspace(1e-2, 800.0, 2000)
    vl = pwlib.gth_local_g(G2, Z, rloc, C)
    ratio = vl / (-4 * np.pi * Z / G2)
    for thr in (0.5, 0.1, 0.01):
        print('nu_loc(G) / (-4 pi Z_ion / G^2) drops below %4.2f at 1/2 G^2 = %5.1f Ht'
              % (thr, 0.5 * G2[np.argmax(ratio < thr)]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(0.5 * G2, 4 * np.pi * 8 / G2, 'k--', label=r'$4\pi Z/G^2$, $Z = 8$ (nucleus)')
    ax[0].loglog(0.5 * G2, 4 * np.pi * Z / G2, 'k:', label=r'$4\pi Z_\mathrm{ion}/G^2$')
    ax[0].loglog(0.5 * G2, -vl, label=r'$-\tilde{\nu}_\mathrm{loc}(G)$ (GTH-PBE)')
    ax[0].set_xlabel(r'$\frac{1}{2}G^2$ [Ht]')
    ax[0].set_ylabel(r'$-\tilde{\nu}(G)$ [Ht bohr$^3$]')
    ax[0].set_ylim(1e-6, 1e4)
    ax[0].legend(fontsize=8)

    # (b') check the analytical transform against a numerical radial
    # transform of the short-range part nu_loc(r) + Z_ion / r
    def short_r(r):
        return pwlib.gth_local_r(r, Z, rloc, C) + Z / r

    for g in (0.5, 2.0, 5.0, 10.0):
        num = 4 * np.pi * quad(lambda r: short_r(r) * np.sin(g * r) / g * r,
                               1e-12, 12.0, limit=200)[0]
        ana = pwlib.gth_local_g(np.array([g**2]), Z, rloc, C)[0] + 4 * np.pi * Z / g**2
        pwlib.check('radial transform at G = %4.1f' % g, num, ana, 1e-7)

    # (c) the potential on the grid of a periodic cell
    # Discarding only the divergent -4 pi Z_ion / G^2 at G = 0 is equivalent to
    # adding a neutralizing background to a lattice of POINT ions; keeping the
    # finite remainder alpha restores everything else (the spread of the ion
    # charge and the short-range terms). The offset is therefore the Madelung
    # term of a point charge, xi Z_ion / a0, and alpha does not appear in it.
    # The grid must be fine enough (spacing ~0.1 bohr) to resolve this.
    print('\n a0    nu_ext(R) - nu_loc(0)   xi Z_ion / a0')
    for a0, N in ((10.0, 128), (15.0, 144), (20.0, 192)):
        M = pwlib.cubic_cell(a0)
        G, G2g = pwlib.gvectors(M, N)
        R = np.full(3, a0 / 2)
        v = pwlib.gth_external_potential(G, G2g, M, [R], [p])
        shift = v[N // 2, N // 2, N // 2] - pwlib.gth_local_r(0.0, Z, rloc, C)
        pred = XI * Z / a0
        print('%4.0f   %+.7f             %+.7f' % (a0, shift, pred))
        pwlib.check('offset at the nucleus, a0 = %g' % a0, shift, pred, 1e-5)
        if a0 == 10.0:
            x = np.arange(N) * a0 / N
            ax[1].plot(x, v[:, N // 2, N // 2], 'o-', ms=3, label=r'$\nu_\mathrm{ext}$ on the grid')
            ax[1].plot(x, pwlib.gth_local_r(np.abs(x - 5.0), Z, rloc, C), '--',
                       label=r'$\nu_\mathrm{loc}(r)$, single atom')
            with np.errstate(divide='ignore'):
                ax[1].plot(x, -Z / np.abs(x - 5.0), ':', label=r'$-Z_\mathrm{ion}/r$')
    ax[1].set_ylim(-40, 5)
    ax[1].set_xlabel('x [bohr]')
    ax[1].set_ylabel('potential [Ht]')
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(ls='--', alpha=0.5, which='both')
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'ex7_1_gth_local.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
