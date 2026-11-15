# -*- coding: utf-8 -*-
"""
Exercise 2.2 -- One orbital, two representations                  [Exercise]

1. Write kinetic_energy in pwlib.py.
2. Complete the TODO blocks below and run this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

DATA = Path(__file__).resolve().parent.parent / 'data'


def main():
    #
    # (a) the 1s orbital of hydrogen, as in the demonstration of Section 2
    #
    a0, N = 10.0, 32
    M = pwlib.cubic_cell(a0)
    r = pwlib.realspace_grid(M, N)
    G, G2 = pwlib.gvectors(M, N)
    # TODO: sample psi_1s centred at R = (5, 5, 5), transform it, and compute
    # (i) the norm by numerical integration in real space, (ii) the sum of
    # the squared coefficients, and (iii) the kinetic energy
    norm_r = None
    norm_g = None
    ekin = None
    pwlib.check('Parseval: both norms are identical', norm_r - norm_g, 0.0, 1e-12)
    pwlib.check('norm at N = 32 (primer, Fig. 4)', norm_r, 1.00077013, 1e-8)
    pwlib.check('E_kin at N = 32 (primer, Fig. 4)', ekin, 0.51662733, 1e-8)

    #
    # (b) the 1pi orbitals of CO (HF/STO-3G); see the description for the
    # layout of the data file
    #
    dat = np.load(DATA / 'co_1pi.npz')
    psi5, psi6 = dat['psi']
    a0, N = float(dat['a0']), int(dat['N'])
    M = pwlib.cubic_cell(a0)
    G, G2 = pwlib.gvectors(M, N)
    print('atomic positions (bohr):\n', dat['positions'])

    # TODO: contour plots of both orbitals in the xz-plane through the
    # molecular axis and in the xy-plane halfway between the nuclei.
    # Remember: psi[i, j, k] belongs to x_i, y_j, z_k, while imshow/contourf
    # expect the vertical axis as the FIRST index (hence the .T).

    # TODO: the 2 x 2 overlap matrix, once in real space and once from the
    # plane-wave coefficients
    S_r = None
    S_g = None
    pwlib.check('<5|5> (real space)', S_r[0, 0], 1.0, 1e-4)
    pwlib.check('<6|6> (real space)', S_r[1, 1], 1.0, 1e-4)
    pwlib.check('<5|6> (real space)', S_r[0, 1], 0.0, 1e-10)
    pwlib.check('real and reciprocal overlaps agree',
                np.abs(S_r - S_g).max(), 0.0, 1e-12)

    # TODO: kinetic energies of both orbitals
    T5 = None
    T6 = None
    pwlib.check('E_kin(psi5) on the grid', T5, 2.000888, 1e-6)
    pwlib.check('E_kin(psi5) vs exact', T5, dat['ekin_exact'][0], 1e-3)
    pwlib.check('E_kin(psi5) = E_kin(psi6)', T5 - T6, 0.0, 1e-10)

    #
    # (c) translation only changes the phases
    #
    shift = np.array([3, -5, 7])                # in grid points
    psi_t = np.roll(psi5, shift, axis=(0, 1, 2))
    # TODO: verify that c_t(G) = c(G) exp(-i G.d), with d the displacement in
    # bohr, and that |c_t(G)| and the kinetic energy are unchanged
    plt.show()


if __name__ == '__main__':
    main()
