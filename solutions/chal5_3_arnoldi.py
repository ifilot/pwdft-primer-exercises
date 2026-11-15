# -*- coding: utf-8 -*-
"""
Challenge 5.3 -- Inside the Arnoldi method                          [SOLUTION]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

DATA = Path(__file__).resolve().parent.parent / 'data'
OUTPUT = Path(__file__).resolve().parent / 'output'


def arnoldi(apply_H, q1, m):
    """
    Plain (unrestarted) Arnoldi iteration with re-orthogonalization.

    Returns the m x m projected matrix H_m = Q^dagger H Q and, for every step
    j = 1..m, the sorted Ritz values of the leading j x j block.
    """
    n = q1.size
    Q = np.zeros((m + 1, n), dtype=complex)
    Hm = np.zeros((m + 1, m), dtype=complex)
    Q[0] = q1.ravel() / np.linalg.norm(q1)
    ritz = []
    for j in range(m):
        w = apply_H(Q[j].reshape(q1.shape)).ravel()
        for _ in range(2):                         # Gram-Schmidt, twice
            h = Q[:j + 1].conj() @ w
            w -= h @ Q[:j + 1]
            Hm[:j + 1, j] += h
        Hm[j + 1, j] = np.linalg.norm(w)
        Q[j + 1] = w / Hm[j + 1, j]
        ritz.append(np.linalg.eigvalsh(Hm[:j + 1, :j + 1]))
    return Hm[:m, :m], ritz, Q[:m]


def main():
    dat = np.load(DATA / 'ch4_orbitals.npz')
    a0, N = float(dat['a0']), int(dat['N'])
    M = pwlib.cubic_cell(a0)
    G, G2 = pwlib.gvectors(M, N)
    vext = pwlib.external_potential(G, G2, M, dat['positions'], dat['charges'])
    veff = pwlib.effective_potential(pwlib.density_from_orbitals(dat['C'], M),
                                     vext, G2)
    eps = dat['eigenvalues']

    rng = np.random.default_rng(7)
    q1 = rng.normal(size=G2.shape) + 1j * rng.normal(size=G2.shape)
    m = 300
    Hm, ritz, Q = arnoldi(lambda c: pwlib.apply_hamiltonian(c, veff, G2, M), q1, m)

    # (a) for a Hermitian H, H_m is tridiagonal (Lanczos)
    off = np.abs(np.triu(Hm, 2)).max()
    pwlib.check('H_m is tridiagonal (max |h_ij|, j > i+1)', off, 0.0, 1e-8)
    pwlib.check('Krylov basis stays orthonormal',
                np.abs(Q.conj() @ Q.T - np.eye(m)).max(), 0.0, 1e-10)

    # (b) Ritz values approach the eigenvalues from above
    lowest = np.array([r[0] for r in ritz])
    pwlib.check('lowest Ritz value never increases', np.max(np.diff(lowest)), 0.0, 1e-10)
    # (the reference eigenvalue itself carries an error of ~1e-6)
    pwlib.check('min. of (lowest Ritz value - eps_1) >= 0',
                min(np.min(lowest - eps[0]), 0.0), 0.0, 1e-6)

    # (c) convergence history of the two lowest distinct levels
    steps = np.arange(1, m + 1)
    err1 = np.abs(lowest - eps[0])
    err2 = np.array([abs(r[1] - eps[1]) if len(r) > 1 else np.nan for r in ritz])
    for tol in (1e-3, 1e-6):
        k1 = steps[np.argmax(err1 < tol)] if np.any(err1 < tol) else None
        k2 = steps[np.argmax(err2 < tol)] if np.any(err2 < tol) else None
        print('steps to reach an error < %.0e: 1a1 (core) %s, 2a1 (valence) %s'
              % (tol, k1, k2))
    spread = np.ptp(ritz[-1])
    print('spread of the spectrum (largest - smallest Ritz value): %.1f Ht' % spread)

    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    for k in range(5):
        x = [j + 1 for j, r in enumerate(ritz) if len(r) > k]
        y = [r[k] for r in ritz if len(r) > k]
        ax[0].plot(x, y, lw=1)
    for e in np.unique(np.round(eps, 6)):
        ax[0].axhline(e, ls='--', color='k', lw=0.6)
    ax[0].set_ylim(-9.5, 1.0)
    ax[0].set_ylabel(r'Ritz values $\theta$ [Ht]')
    ax[1].semilogy(steps, err1, label=r'$1a_1$')
    ax[1].semilogy(steps, err2, label=r'$2a_1$')
    ax[1].set_xlabel('Arnoldi step $m$')
    ax[1].set_ylabel(r'$|\theta - \epsilon|$ [Ht]')
    ax[1].legend()
    for a in ax:
        a.grid(ls='--', alpha=0.5)
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'chal5_3_arnoldi.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
