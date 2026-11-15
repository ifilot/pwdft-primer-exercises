# -*- coding: utf-8 -*-
"""
Exercise 5.1 -- The total energy of methane from a set of orbitals [SOLUTION]
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

    # (a) orthonormality and the number of electrons
    S = np.array([[np.vdot(a, b) for b in C] for a in C])
    pwlib.check('orbitals are orthonormal', np.abs(S - np.eye(len(C))).max(),
                0.0, 1e-8)
    rho = pwlib.density_from_orbitals(C, M)
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

    # (c) the sum of orbital energies is not the total energy
    eps = dat['eigenvalues']
    print('2 * sum of orbital energies = %.6f Ht (total energy %.6f Ht)'
          % (2 * eps.sum(), E['total']))
    vxc = pwlib.lda_xc(rho)[1]
    corr = (-E['hartree'] - pwlib.integrate(vxc * rho, M) + E['xc'] + eewald)
    pwlib.check('2 sum eps - E_H - int v_xc rho + E_xc + E_Ewald = E_total',
                2 * eps.sum() + corr, E['total'], 1e-5)

    # (d) the electrostatic terms only make sense together
    print('E_nuc + E_H + E_Ewald = %.6f Ht' % (E['nuclear'] + E['hartree'] + eewald))


if __name__ == '__main__':
    main()
