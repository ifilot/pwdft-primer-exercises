# -*- coding: utf-8 -*-
"""
Challenge 5.6 -- A real basis for degenerate orbitals                  [Challenge]

Outline only; see the description for the questions to answer.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

DATA = Path(__file__).resolve().parent.parent / 'data'


def real_basis(psis):
    """
    TODO: a real orthonormal basis for the span of the degenerate complex
    orbitals psis. Consider the 2n real functions Re psi_k and Im psi_k:
    why do they lie in the span, and how many of them are independent?
    Hint: np.linalg.svd.
    """
    raise NotImplementedError


def main():
    dat = np.load(DATA / 'ch4_orbitals.npz')
    # the t2 orbitals are dat['C'][2:5]
    # verify that the new orbitals are (i) real, (ii) orthonormal, (iii)
    # eigenfunctions with eigenvalue eps_t2, and (iv) give the same density
    # optional: rotate them onto functions that resemble p_x, p_y and p_z
    plt.show()


if __name__ == '__main__':
    main()
