# -*- coding: utf-8 -*-
"""
Exercise 5.2 -- Applying the Hamiltonian without storing it        [SOLUTION]
"""

import time
from pathlib import Path

import numpy as np

import pwlib

DATA = Path(__file__).resolve().parent.parent / 'data'


def dense_hamiltonian(veff, G2, M):
    """Build the full matrix by applying H to every unit vector."""
    n = G2.size
    H = np.empty((n, n), dtype=complex)
    e = np.zeros(G2.shape, dtype=complex)
    for k in range(n):
        e.flat[k] = 1.0
        H[:, k] = pwlib.apply_hamiltonian(e, veff, G2, M).ravel()
        e.flat[k] = 0.0
    return H


def main():
    rng = np.random.default_rng(42)

    # (a) Hermiticity on the CH4 grid with the reference potential
    dat = np.load(DATA / 'ch4_orbitals.npz')
    a0, N = float(dat['a0']), int(dat['N'])
    M = pwlib.cubic_cell(a0)
    G, G2 = pwlib.gvectors(M, N)
    vext = pwlib.external_potential(G, G2, M, dat['positions'], dat['charges'])
    rho = pwlib.density_from_orbitals(dat['C'], M)
    veff = pwlib.effective_potential(rho, vext, G2)
    x = rng.normal(size=G2.shape) + 1j * rng.normal(size=G2.shape)
    y = rng.normal(size=G2.shape) + 1j * rng.normal(size=G2.shape)
    lhs = np.vdot(x, pwlib.apply_hamiltonian(y, veff, G2, M))
    rhs = np.vdot(pwlib.apply_hamiltonian(x, veff, G2, M), y)
    pwlib.check('<x|Hy> = <Hx|y>', abs(lhs - rhs) / abs(lhs), 0.0, 1e-12)

    # (b) the reference orbitals are eigenvectors of H[rho]
    for i, (c, e) in enumerate(zip(dat['C'], dat['eigenvalues'])):
        res = np.linalg.norm(pwlib.apply_hamiltonian(c, veff, G2, M) - e * c)
        print('orbital %i: eps = %+.6f Ht, residual |Hc - eps c| = %.1e' % (i + 1, e, res))
    t0 = time.perf_counter()
    eps, Cn = pwlib.lowest_eigenstates_arpack(veff, G2, M, 5)
    print('eigsh, N = 32 (%i plane waves): %.2f s' % (G2.size, time.perf_counter() - t0))
    pwlib.check('eigsh reproduces the orbital energies',
                np.abs(eps - dat['eigenvalues']).max(), 0.0, 1e-5)

    # (c) dense versus iterative diagonalization on small grids, using the
    # potential of a hydrogen atom
    print('\n  N    N_pw   dense eigh [s]   eigsh [s]   block [s]   '
          'max|dense - eigsh|  max|dense - block|')
    for Nh in (8, 10, 12, 14):
        Mh = pwlib.cubic_cell(10.0)
        Gh, G2h = pwlib.gvectors(Mh, Nh)
        vh = pwlib.external_potential(Gh, G2h, Mh, [[5.0, 5.0, 5.0]], [1.0])
        H = dense_hamiltonian(vh, G2h, Mh)
        t0 = time.perf_counter()
        ed = np.linalg.eigvalsh(H)[:5]
        td = time.perf_counter() - t0
        t0 = time.perf_counter()
        ei, _ = pwlib.lowest_eigenstates_arpack(vh, G2h, Mh, 5)
        ti = time.perf_counter() - t0
        t0 = time.perf_counter()
        eb, _ = pwlib.lowest_eigenstates(vh, G2h, Mh, 5)
        tb = time.perf_counter() - t0
        print('%3i  %6i   %10.3f     %8.3f    %8.3f      %.1e           %.1e'
              % (Nh, Nh**3, td, ti, tb, np.abs(ed - ei).max(),
                 np.abs(ed - eb).max()))
    # the dense diagonalization scales as N_pw^3 = N^9
    print('extrapolated dense diagonalization for N = 32: %.0f s (%.1f h), '
          'matrix of %.0f GB' % (td * (32 / 14)**9, td * (32 / 14)**9 / 3600,
                                  (32**3)**2 * 16 / 1e9))
    print('dense (N = 14):', np.round(ed, 6))
    print('eigsh (N = 14):', np.round(ei, 6))
    print('block (N = 14):', np.round(eb, 6))
    pwlib.check('block solver finds all three p-like states', np.abs(ed - eb).max(),
                0.0, 1e-6)


if __name__ == '__main__':
    main()
