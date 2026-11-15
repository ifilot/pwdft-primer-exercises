# -*- coding: utf-8 -*-
"""
Exercise 5.1 -- The total energy of methane from a set of orbitals [Exercise]

1. Write density_from_orbitals and energy_components in pwlib.py.
2. Complete the TODO blocks below and run this script.
"""

from pathlib import Path

import numpy as np

import pwlib

DATA = Path(__file__).resolve().parent.parent / 'data'


def main():
    dat = np.load(DATA / 'ch4_orbitals.npz')
    C = dat['C']                               # (5, 32, 32, 32) coefficients
    a0, N = float(dat['a0']), int(dat['N'])
    positions, charges = dat['positions'], dat['charges']
    M = pwlib.cubic_cell(a0)
    G, G2 = pwlib.gvectors(M, N)

    # (a) TODO: the overlap matrix of the orbitals (from the coefficients)
    # and the number of electrons
    S = None
    rho = None
    pwlib.check('orbitals are orthonormal', np.abs(S - np.eye(len(C))).max(),
                0.0, 1e-8)
    pwlib.check('number of electrons', pwlib.integrate(rho, M), 10.0, 1e-8)

    # (b) the energy terms
    vext = pwlib.external_potential(G, G2, M, positions, charges)
    eewald = pwlib.ewald_energy(M, positions, charges)
    E = pwlib.energy_components(C, M, G2, vext, eewald)
    for k, v in E.items():
        print('  %-8s %+.6f Ht' % (k, v))
    ref = {'kinetic': 31.907966, 'nuclear': -81.735803, 'hartree': 19.038083,
           'xc': -6.284507, 'ewald': -0.481269, 'total': -37.555531}
    for k in ref:
        pwlib.check('E_%s' % k, E[k], ref[k], 2e-6)

    # (c) TODO: compare 2 * sum of the orbital energies (dat['eigenvalues'])
    # with the total energy, and reconstruct the total energy from the
    # orbital energies by correcting for double counting

    # (d) TODO: print E_nuc + E_H + E_Ewald


if __name__ == '__main__':
    main()
