# -*- coding: utf-8 -*-
"""
Exercise 6.2 -- The cell size, absolute orbital energies and box states [SOLUTION]

Requires PyPWDFT >= 0.3.0.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pypwdft import PWDFT

import pwlib

OUTPUT = Path(__file__).resolve().parent / 'output'


def main():
    ecut = 40
    Ls = np.array([8.0, 10.0, 12.0, 15.0, 20.0])
    E, eps, face = [], [], []
    for L in Ls:
        res = PWDFT(pwlib.diatomic('H2', L), cutoff=ecut, xc='pbe',
                    pseudopotential='all-electron').run(bands=5)
        E.append(res.energy.total)
        eps.append(res.orbitals.energies)
        rho = pwlib.from_pypwdft(res.density)
        face.append(max(rho[0].max(), rho[:, 0].max(), rho[:, :, 0].max()))
        print('L = %4.1f: E = %.6f Ht, eps = %s, max. density on the cell face = %.1e'
              % (L, E[-1], np.round(eps[-1], 4), face[-1]))
    E, eps = np.array(E), np.array(eps)

    # (a) total energy
    print('\nE(L) - E(20) [mHt]:', np.round(1000 * (E - E[-1]), 2))
    pwlib.check('E(L=8) below E(L=20) by about 11 mHt (primer, at 160 Ht)',
                1000 * (E[0] - E[-1]), -11.0, 3.0)

    # (b) occupied level: absolute value versus cell size
    print('HOMO [Ht]:', np.round(eps[:, 0], 4))
    print('change from L = 12 to 20: HOMO %.1f mHt, total energy %.2f mHt'
          % (1000 * (eps[-1, 0] - eps[2, 0]), 1000 * (E[-1] - E[2])))

    # (c) unoccupied states: box states whose energies approach zero as the
    # cell grows; for the higher ones eps * L^2 is roughly constant (particle
    # in a box), the lowest one does not follow this law
    for k in range(1, eps.shape[1]):
        print('eps_%d * L^2:' % (k + 1), np.round(eps[:, k] * Ls**2, 2))
    pwlib.check('box states 4 and 5 are degenerate (all L)',
                np.abs(eps[:, 3] - eps[:, 4]).max(), 0.0, 1e-4)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(Ls, 1000 * (E - E[-1]), 'o-')
    ax[0].set_xlabel('L [bohr]')
    ax[0].set_ylabel(r'$E(L) - E(20)$ [mHt]')
    for k in range(eps.shape[1]):
        ax[1].plot(1.0 / Ls**2, eps[:, k], 'o-', label=r'$\epsilon_{%d}$' % (k + 1))
    ax[1].set_xlabel(r'$1/L^2$ [bohr$^{-2}$]')
    ax[1].set_ylabel('orbital energy [Ht]')
    ax[1].legend()
    for a in ax:
        a.grid(ls='--', alpha=0.5)
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'ex6_2_cell_size.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
