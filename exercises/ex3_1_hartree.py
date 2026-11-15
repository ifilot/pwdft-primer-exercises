# -*- coding: utf-8 -*-
"""
Exercise 3.1 -- The Hartree potential of a Gaussian charge cloud   [Exercise]

1. Write hartree_potential in pwlib.py.
2. Complete the TODO blocks below and run this script.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

import pwlib

XI = 2.837297       # Madelung constant, simple cubic, neutralizing background


def erep_periodic(a0, r2=0.75):
    """TODO: expected repulsion energy of the periodic system (Section 3)."""
    raise NotImplementedError


def main():
    a0 = 10.0
    M = pwlib.cubic_cell(a0)
    centre = np.array([5.0, 5.0, 5.0])

    # (a) TODO: compute E_rep = int v_H rho for N = 16, 24, 32, 48 and print
    # the deviation from erep_periodic(a0). Use pwlib.gaussian_density(r, M,
    # centre, alpha=2.0) for rho = |g|^2.
    erep = None          # value for N = 48
    vh, r, N = None, None, 48

    pwlib.check('E_rep vs eq. (erep_periodic), N = 48', erep,
                erep_periodic(a0), 1e-6)
    pwlib.check('E_H = E_rep / 2 vs primer Table (L = 10)', 0.5 * erep,
                0.4239, 5e-5)

    # (b) the potential itself
    pwlib.check('cell average of v_H vanishes', np.mean(vh), 0.0, 1e-12)
    x = r[:, N // 2, N // 2, 0]
    d = np.abs(x - 5.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        v_iso = np.where(d > 0, erf(np.sqrt(2.0) * d) / d, 2 * np.sqrt(2 / np.pi))
    # TODO: plot v_H and v_iso along this line and determine their
    # difference at the centre of the cloud
    plt.show()


if __name__ == '__main__':
    main()
