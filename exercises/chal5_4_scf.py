# -*- coding: utf-8 -*-
"""
Challenge 5.4 -- Your own self-consistent field program                [Challenge]

Skeleton of the SCF cycle; see the description for the questions to answer.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

CH4 = (np.array([[5.0000, 5.0000, 5.0000],
                 [6.1958, 6.1958, 6.1958],
                 [3.8042, 3.8042, 6.1958],
                 [3.8042, 6.1958, 3.8042],
                 [6.1958, 3.8042, 3.8042]]),
       np.array([6.0, 1.0, 1.0, 1.0, 1.0]))
H2 = (np.array([[5.0, 5.0, 4.3], [5.0, 5.0, 5.7]]), np.array([1.0, 1.0]))


def scf(a0, N, positions, charges, beta=0.5, etol=1e-7, rhotol=1e-6,
        maxiter=100, verbose=True):
    """
    Minimal Kohn-Sham SCF cycle: Gamma-point, bare Coulomb nuclei, LDA,
    all plane waves on the grid as basis, linear density mixing.
    """
    M = pwlib.cubic_cell(a0)
    G, G2 = pwlib.gvectors(M, N)
    nelec = int(round(np.sum(charges)))
    nocc = nelec // 2
    # TODO: quantities that do not change during the cycle
    vext = None
    eewald = None

    # TODO: initial guess, a homogeneous density with nelec electrons
    rho = None
    C, E_old = None, 0.0
    for it in range(1, maxiter + 1):
        # TODO:
        #  1. effective potential from rho
        #  2. the nocc lowest eigenstates (pwlib.lowest_eigenstates, starting
        #     from the previous orbitals C0=C)
        #  3. new density and total energy (pwlib.energy_components)
        #  4. convergence test on |dE| AND rms(rho_new - rho)
        #  5. linear mixing
        raise NotImplementedError
    raise RuntimeError('SCF did not converge in %i iterations' % maxiter)


def main():
    # (a) H2: expected total energy -1.1373616 Ht
    # (b) CH4: compare with the reference data of Exercise 5.1,
    #     np.load('../data/ch4_orbitals.npz'): total_energy and eigenvalues
    # (c) number of iterations versus beta for CH4 (a0 = 10, N = 32), and
    #     for H2 in cells of 10, 15 and 20 bohr (N = 32, 48, 64)
    plt.show()


if __name__ == '__main__':
    main()
