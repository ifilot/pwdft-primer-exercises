# -*- coding: utf-8 -*-
"""
Exercise 5.2 -- Applying the Hamiltonian without storing it        [Exercise]

1. Write effective_potential, apply_hamiltonian, hamiltonian_operator and
   lowest_eigenstates_arpack in pwlib.py (lowest_eigenstates is provided).
2. Complete the TODO blocks below and run this script.
"""

import time
from pathlib import Path

import numpy as np

import pwlib

DATA = Path(__file__).resolve().parent.parent / 'data'


def dense_hamiltonian(veff, G2, M):
    """TODO: build the full N_pw x N_pw matrix by applying H to every unit
    vector (only feasible for small grids!)."""
    raise NotImplementedError


def main():
    rng = np.random.default_rng(42)

    # (a) Hermiticity on the CH4 grid with the potential of the reference
    # orbitals
    dat = np.load(DATA / 'ch4_orbitals.npz')
    a0, N = float(dat['a0']), int(dat['N'])
    M = pwlib.cubic_cell(a0)
    G, G2 = pwlib.gvectors(M, N)
    vext = pwlib.external_potential(G, G2, M, dat['positions'], dat['charges'])
    rho = pwlib.density_from_orbitals(dat['C'], M)
    veff = pwlib.effective_potential(rho, vext, G2)
    # TODO: for two random complex vectors x and y, compare <x|Hy> and <Hx|y>
    lhs, rhs = None, None
    pwlib.check('<x|Hy> = <Hx|y>', abs(lhs - rhs) / abs(lhs), 0.0, 1e-12)

    # (b) TODO: print the residual |Hc - eps c| of every reference orbital,
    # then compute the five lowest eigenpairs with your
    # lowest_eigenstates_arpack and compare with dat['eigenvalues']

    # (c) TODO: for the hydrogen atom (proton at (5, 5, 5), a0 = 10) on grids
    # N = 8, 10, 12, 14: compare the five lowest eigenvalues and the timings
    # of dense diagonalization (np.linalg.eigvalsh), lowest_eigenstates_arpack
    # and the provided block solver lowest_eigenstates.


if __name__ == '__main__':
    main()
