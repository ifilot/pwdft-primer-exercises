# -*- coding: utf-8 -*-
"""
Challenge 5.3 -- Inside the Arnoldi method                             [Challenge]

Outline only; see the description for the questions to answer.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

DATA = Path(__file__).resolve().parent.parent / 'data'


def arnoldi(apply_H, q1, m):
    """
    TODO: plain (unrestarted) Arnoldi iteration. Return the m x m matrix H_m
    and, for every step j = 1..m, the sorted Ritz values of its leading
    j x j block. Orthogonalize every new vector against ALL previous ones
    (doing it twice is cheap insurance against rounding errors).
    """
    raise NotImplementedError


def main():
    dat = np.load(DATA / 'ch4_orbitals.npz')
    a0, N = float(dat['a0']), int(dat['N'])
    M = pwlib.cubic_cell(a0)
    G, G2 = pwlib.gvectors(M, N)
    vext = pwlib.external_potential(G, G2, M, dat['positions'], dat['charges'])
    veff = pwlib.effective_potential(pwlib.density_from_orbitals(dat['C'], M),
                                     vext, G2)
    eps = dat['eigenvalues']
    # run m = 300 steps from a random start vector and
    #   (a) check that H_m is tridiagonal
    #   (b) check that the lowest Ritz value decreases monotonically and stays
    #       above eps_1
    #   (c) plot the Ritz values and the errors of the 1a1 and 2a1 levels
    plt.show()


if __name__ == '__main__':
    main()
