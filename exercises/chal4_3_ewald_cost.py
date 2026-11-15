# -*- coding: utf-8 -*-
"""
Challenge 4.3 -- The cost of an Ewald sum and the optimal split        [Challenge]

Outline only; see the description for the questions to answer.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

import pwlib

M_NACL = 1.747565


def ewald_fast(M, positions, charges, gcut=2.0, gamma=1e-8):
    """
    TODO: a vectorized Ewald sum. Evaluate the reciprocal-space part with the
    structure factor S(G) = sum_j q_j exp(i G.R_j) (derive why this works).
    Return (E, number of lattice vectors, number of G-vectors).
    """
    raise NotImplementedError


def nacl_supercell(a, n):
    """n x n x n supercell of the conventional rock-salt cell (edge a)."""
    na = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]) * a / 2
    cl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]) * a / 2
    shifts = pwlib.integer_vectors([n - 1] * 3, include_zero=True)
    shifts = shifts[np.all(shifts >= 0, axis=1)] * a
    pos = np.vstack([(np.vstack([na, cl]) + s) for s in shifts])
    q = np.tile([1.0] * 4 + [-1.0] * 4, len(shifts))
    return pwlib.cubic_cell(n * a), pos, q


def main():
    # (a) validate ewald_fast against pwlib.ewald_energy (CO) and against the
    #     Madelung constant of NaCl (2 x 2 x 2 supercell, a = 10.66 bohr)
    # (b) wall time versus G_cut for CO (a0 = 10 and 20) and NaCl 2 x 2 x 2;
    #     use time.perf_counter() and take the best of a few repetitions
    plt.show()


if __name__ == '__main__':
    main()
