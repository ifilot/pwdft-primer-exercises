# -*- coding: utf-8 -*-
"""
Exercise 4.2 -- Why the background charge does no harm             [Exercise]

Uses hartree_potential, external_potential and ewald_energy from pwlib.py.
Complete the TODO block below and run this script.
"""

import numpy as np

import pwlib

XI = 2.837297


def electrostatics(L, N):
    """
    TODO: return (E_H, E_nuc, E_Ewald) for a normalized Gaussian electron
    density (alpha = 2) centred on a proton at the centre of a cubic cell of
    edge L, sampled with N points per direction. Note E_H = 1/2 int v_H rho.
    """
    raise NotImplementedError


def main():
    table = {10: (0.4239, -1.3136, -0.1419), 20: (0.4935, -1.4541, -0.0709)}
    iso = (1 / np.sqrt(np.pi), -2 * np.sqrt(2 / np.pi), 0.0)
    print('  L      E_H       E_nuc     E_Ewald     Sum')
    for L, N in ((10, 48), (20, 96)):
        e = electrostatics(L, N)
        print('%4i  %8.4f  %8.4f  %8.4f  %9.4f' % (L, *e, sum(e)))
        for name, val, ref in zip(('E_H', 'E_nuc', 'E_Ewald'), e, table[L]):
            pwlib.check('%s, L = %i (primer, Table)' % (name, L), val, ref, 5e-5)
        pwlib.check('E_Ewald = -xi / (2L), L = %i' % L, e[2], -XI / (2 * L), 1e-6)
        pwlib.check('sum = isolated value, L = %i' % L, sum(e), sum(iso), 1e-6)
    print('isolated: %8.4f  %8.4f  %8.4f  %9.4f' % (*iso, sum(iso)))


if __name__ == '__main__':
    main()
