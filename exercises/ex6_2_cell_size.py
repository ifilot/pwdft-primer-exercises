# -*- coding: utf-8 -*-
"""
Exercise 6.2 -- The cell size, absolute orbital energies and box states [Exercise]

Requires PyPWDFT >= 0.3.0. Complete the TODO blocks and run this script
(about two minutes).
"""

import matplotlib.pyplot as plt
import numpy as np
from pypwdft import PWDFT

import pwlib


def main():
    ecut = 40
    Ls = np.array([8.0, 10.0, 12.0, 15.0, 20.0])
    E, eps, face = [], [], []
    for L in Ls:
        res = PWDFT(pwlib.diatomic('H2', L), cutoff=ecut, xc='pbe',
                    pseudopotential='all-electron').run(bands=5)
        E.append(res.energy.total)
        eps.append(res.orbitals.energies)
        # PyPWDFT stores grids as [z, y, x]; convert to the pwlib convention
        rho = pwlib.from_pypwdft(res.density)
        # TODO: the largest density on the faces of the cell
        face.append(None)
    E, eps = np.array(E), np.array(eps)

    # (a) total energy
    print('E(L) - E(20) [mHt]:', np.round(1000 * (E - E[-1]), 2))
    pwlib.check('E(L=8) below E(L=20) by about 11 mHt (primer, at 160 Ht)',
                1000 * (E[0] - E[-1]), -11.0, 3.0)

    # (b) TODO: how much does the HOMO change between L = 12 and 20, compared
    # with the total energy?

    # (c) TODO: plot all orbital energies against 1/L^2 and print eps * L^2
    # for the unoccupied states
    plt.show()


if __name__ == '__main__':
    main()
