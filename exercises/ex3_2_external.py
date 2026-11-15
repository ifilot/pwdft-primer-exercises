# -*- coding: utf-8 -*-
"""
Exercise 3.2 -- The external potential and cancelling Madelung terms [Exercise]

1. Write external_potential in pwlib.py.
2. Complete the TODO blocks below and run this script.
"""

import numpy as np

import pwlib

XI = 2.837297


def epot_periodic(a0, r2=0.75):
    """TODO: expected nuclear attraction of the Gaussian cloud (Section 3)."""
    raise NotImplementedError


def energies(a0, N, R, flip_sign=False):
    """
    E_nuc and E_rep of a Gaussian cloud (alpha = 2) centred on a proton at R.
    With flip_sign=True the structure factor is deliberately evaluated with
    the wrong sign, exp(+iG.R), by passing -G to external_potential.
    """
    M = pwlib.cubic_cell(a0)
    r = pwlib.realspace_grid(M, N)
    G, G2 = pwlib.gvectors(M, N)
    rho = pwlib.gaussian_density(r, M, R, alpha=2.0)
    vext = pwlib.external_potential(-G if flip_sign else G, G2, M, [R], [1.0])
    vh = pwlib.hartree_potential(rho, G2)
    return pwlib.integrate(vext * rho, M), pwlib.integrate(vh * rho, M)


def main():
    a0, N = 10.0, 48
    R = np.array([3.7, 5.2, 6.4])          # deliberately off-centre and off-grid

    # (a) nuclear attraction
    enuc, erep = energies(a0, N, R)
    pwlib.check('E_nuc vs the periodic-system formula (Section 3)', enuc, epot_periodic(a0), 1e-6)
    pwlib.check('E_nuc vs Table in Section 4 (L = 10)', enuc, -1.3136, 5e-5)

    # (b) TODO: show that E_nuc + E_rep deviates from the isolated value
    # 2 (1 - sqrt 2) / sqrt(pi) by only 2 pi <r^2> / (3 a0^3), for a0 = 10
    # (N = 48) and a0 = 20 (N = 96)

    # (c) TODO: compare E_nuc with the correct and with the flipped sign, for
    # the off-centre proton and for a proton at the cell centre (5, 5, 5)


if __name__ == '__main__':
    main()
