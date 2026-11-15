# -*- coding: utf-8 -*-
"""
Challenge 4.3 -- The cost of an Ewald sum and the optimal split      [SOLUTION]
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

import pwlib

OUTPUT = Path(__file__).resolve().parent / 'output'
M_NACL = 1.747565


def ewald_fast(M, positions, charges, gcut=2.0, gamma=1e-8):
    """
    Vectorized Ewald sum. The reciprocal-space double sum over atom pairs is
    replaced by the squared modulus of the structure factor,
        sum_ij q_i q_j cos(G.R_ij) = |S(G)|^2,  S(G) = sum_j q_j exp(i G.R_j),
    which reduces its cost from O(N_atom^2 N_G) to O(N_atom N_G).
    """
    R = np.asarray(positions, dtype=float)
    q = np.asarray(charges, dtype=float)
    Omega = pwlib.cell_volume(M)
    alpha, nmax, mmax = pwlib.ewald_parameters(M, gcut, gamma)

    # real space: loop over atoms i, vectorize over atoms j and cells
    T = pwlib.integer_vectors(nmax, include_zero=True) @ M
    sa = np.sqrt(alpha)
    Esr = 0.0
    for i in range(len(q)):
        d = np.linalg.norm((R[i] - R)[:, None, :] + T[None, :, :], axis=-1)
        d[i, np.all(T == 0, axis=1)] = np.inf          # exclude self-term
        Esr += 0.5 * q[i] * np.sum(q[:, None] * erfc(sa * d) / d)

    # reciprocal space via the structure factor
    B = 2.0 * np.pi * np.linalg.inv(M.T)
    G = pwlib.integer_vectors(mmax) @ B
    G2 = np.einsum('ij,ij->i', G, G)
    S = np.exp(1j * G @ R.T) @ q
    Elr = 2.0 * np.pi / Omega * np.sum(np.exp(-0.25 * G2 / alpha) / G2
                                       * np.abs(S)**2)

    Es = np.sqrt(alpha / np.pi) * np.sum(q**2)
    Een = np.pi / (2.0 * alpha * Omega) * np.sum(q)**2
    return Esr + Elr - Es - Een, len(T), len(G)


def nacl_supercell(a, n):
    """n x n x n supercell of the conventional rock-salt cell (edge a)."""
    na = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]) * a / 2
    cl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]) * a / 2
    shifts = pwlib.integer_vectors([n - 1] * 3, include_zero=True)
    shifts = shifts[np.all(shifts >= 0, axis=1)] * a
    pos = np.vstack([(np.vstack([na, cl]) + s) for s in shifts])
    q = np.tile([1.0] * 4 + [-1.0] * 4, len(shifts))
    return pwlib.cubic_cell(n * a), pos, q


def timing(fun, *args, repeat=3, **kw):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fun(*args, **kw)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    # (a) validation against the straightforward implementation
    M = pwlib.cubic_cell(10.0)
    pos = np.array([[5.0, 5.0, 5.96769844], [5.0, 5.0, 3.70973478]])
    q = np.array([8.0, 6.0])
    pwlib.check('fast vs reference implementation (CO)',
                ewald_fast(M, pos, q)[0], pwlib.ewald_energy(M, pos, q), 1e-10)
    a = 10.66                                    # NaCl lattice constant (bohr)
    Ms, ps, qs = nacl_supercell(a, 2)
    E = ewald_fast(Ms, ps, qs)[0]
    pwlib.check('Madelung constant from a 2x2x2 NaCl supercell',
                -E / (len(qs) / 2) * (a / 2), M_NACL, 1e-6)

    # (b) cost versus G_cut for three systems
    systems = {
        'CO, a0 = 10': (M, pos, q),
        'CO, a0 = 20': (pwlib.cubic_cell(20.0), pos + 5.0, q),
        'NaCl 2x2x2 (64 ions)': (Ms, ps, qs),
    }
    gcuts = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (Mx, px, qx) in systems.items():
        times = np.array([timing(ewald_fast, Mx, px, qx, gcut=g)[0] for g in gcuts])
        best = gcuts[np.argmin(times)]
        print('%-22s optimal G_cut = %.2f (t = %.2e s); at G_cut = 2: %.2e s'
              % (name, best, times.min(), times[gcuts == 2.0][0]))
        ax.loglog(gcuts, times, 'o-', label=name)
    ax.set_xlabel(r'$G_\mathrm{cut}$ [bohr$^{-1}$]')
    ax.set_ylabel('wall time [s]')
    ax.legend()
    ax.grid(ls='--', alpha=0.5, which='both')
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'chal4_3_ewald_cost.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
