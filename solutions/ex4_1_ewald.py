# -*- coding: utf-8 -*-
"""
Exercise 4.1 -- The four terms of the Ewald sum                    [SOLUTION]
"""

import numpy as np

import pwlib

XI = 2.837297           # simple cubic lattice in a neutralizing background
M_NACL = 1.747565       # Madelung constant of rock salt


def nacl_cell(a):
    """Conventional cubic rock-salt cell with edge a (nearest neighbours a/2)."""
    na = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]) * a / 2
    cl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]) * a / 2
    return np.vstack([na, cl]), np.array([1.0] * 4 + [-1.0] * 4)


def main():
    a0 = 10.0
    M = pwlib.cubic_cell(a0)

    # (a) the parameter recipe of the primer
    for gcut, ref in ((2.0, (3, 5)), (0.1, (28, 2)), (10.0, (2, 17))):
        _, nmax, mmax = pwlib.ewald_parameters(M, gcut)
        print('G_cut = %5.1f: n_max = %2i, m_max = %2i   (primer: %i, %i)'
              % (gcut, nmax[0], mmax[0], *ref))

    # (b) one unit charge in a cubic cell: E = -xi / (2 a0)
    E1 = pwlib.ewald_energy(M, [[1.0, 2.0, 3.0]], [1.0])
    pwlib.check('Madelung constant xi = -2 a0 E', -2 * a0 * E1, XI, 1e-6)

    # (c) rock salt: E per ion pair = -M_NaCl / d
    pos, q = nacl_cell(a0)
    E, terms = pwlib.ewald_energy(M, pos, q, return_terms=True)
    pwlib.check('Madelung constant of NaCl', -E / 4 * (a0 / 2), M_NACL, 1e-6)
    pwlib.check('E_en vanishes for a neutral cell', terms['en'], 0.0, 1e-12)

    # (d) CO: independence of the splitting parameter
    pos = np.array([[5.0, 5.0, 5.96769844], [5.0, 5.0, 3.70973478]])
    q = np.array([8.0, 6.0])
    print('\nG_cut   alpha      E_sr        E_lr        E_s         E_en        E_Ewald')
    energies = []
    for gcut in (0.5, 1.0, 2.0, 4.0, 6.0):
        E, t = pwlib.ewald_energy(M, pos, q, gcut=gcut, return_terms=True)
        alpha, _, _ = pwlib.ewald_parameters(M, gcut)
        energies.append(E)
        print('%4.1f  %8.5f  %10.5f  %10.5f  %10.5f  %10.5f  %12.8f'
              % (gcut, alpha, t['sr'], t['lr'], t['s'], t['en'], E))
    pwlib.check('spread of E_Ewald over G_cut', np.ptp(energies), 0.0, 1e-8)
    pwlib.check('E_Ewald of CO in a 10 bohr cell', energies[2], -5.99562232, 1e-7)

    # (e) the same nuclei without background: the bare pair repulsion
    print('bare C-O repulsion Z_C Z_O / d = %.5f Ht'
          % (48.0 / np.linalg.norm(pos[0] - pos[1])))


if __name__ == '__main__':
    main()
