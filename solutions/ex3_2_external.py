# -*- coding: utf-8 -*-
"""
Exercise 3.2 -- The external potential and cancelling Madelung terms [SOLUTION]
"""

import numpy as np

import pwlib

XI = 2.837297


def epot_periodic(a0, r2=0.75):
    """Expected nuclear attraction of the Gaussian cloud (eq. epot_periodic)."""
    return -2.0 * np.sqrt(2.0 / np.pi) + XI / a0 - 2.0 * np.pi * r2 / (3.0 * a0**3)


def energies(a0, N, R, flip_sign=False):
    M = pwlib.cubic_cell(a0)
    r = pwlib.realspace_grid(M, N)
    G, G2 = pwlib.gvectors(M, N)
    rho = pwlib.gaussian_density(r, M, R, alpha=2.0)
    if flip_sign:   # deliberately wrong structure factor exp(+iG.R)
        vext = pwlib.external_potential(-G, G2, M, [R], [1.0])
    else:
        vext = pwlib.external_potential(G, G2, M, [R], [1.0])
    vh = pwlib.hartree_potential(rho, G2)
    return pwlib.integrate(vext * rho, M), pwlib.integrate(vh * rho, M)


def main():
    a0, N = 10.0, 48
    R = np.array([3.7, 5.2, 6.4])          # deliberately off-centre and off-grid

    # (a) nuclear attraction
    enuc, erep = energies(a0, N, R)
    pwlib.check('E_nuc vs eq. (epot_periodic)', enuc, epot_periodic(a0), 1e-6)
    pwlib.check('E_nuc vs primer Table (L = 10)', enuc, -1.3136, 5e-5)

    # (b) the Madelung terms cancel in the sum
    eiso = 2.0 * (1.0 - np.sqrt(2.0)) / np.sqrt(np.pi)
    pwlib.check('E_nuc + E_rep - E_iso = 2 pi <r^2> / (3 a0^3)',
                enuc + erep - eiso, 2 * np.pi * 0.75 / (3 * a0**3), 1e-6)
    enuc20, erep20 = energies(20.0, 96, R + 5.0)
    pwlib.check('same for a0 = 20', enuc20 + erep20 - eiso,
                2 * np.pi * 0.75 / (3 * 20.0**3), 1e-6)

    # (c) a sign error in the structure factor
    for label, Rt in (('off-centre', R), ('cell centre', np.array([5.0, 5.0, 5.0]))):
        good, _ = energies(a0, N, Rt)
        bad, _ = energies(a0, N, Rt, flip_sign=True)
        print('%-12s  E_nuc correct = %+.6f   with exp(+iG.R) = %+.6f'
              % (label, good, bad))


if __name__ == '__main__':
    main()
