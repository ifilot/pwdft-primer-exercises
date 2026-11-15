# -*- coding: utf-8 -*-
"""
Challenge 5.6 -- A real basis for degenerate orbitals                [SOLUTION]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

DATA = Path(__file__).resolve().parent.parent / 'data'
OUTPUT = Path(__file__).resolve().parent / 'output'


def real_basis(psis):
    """
    Real orthonormal basis for the span of a set of degenerate complex
    orbitals. Because the Hamiltonian is real, the span is closed under
    complex conjugation, so the real and imaginary parts of the orbitals lie
    in the span as well. Of these 2n real functions only n are linearly
    independent; a singular value decomposition extracts them.
    """
    n = len(psis)
    F = np.vstack([p.real.ravel() for p in psis] + [p.imag.ravel() for p in psis])
    U, s, Vt = np.linalg.svd(F, full_matrices=False)
    print('singular values of [Re psi; Im psi]:', np.round(s / s[0], 8))
    phis = Vt[:n].reshape((n,) + psis[0].shape)
    # normalize with respect to the grid integral (the rows of Vt have unit
    # Euclidean norm); the sign of each function is arbitrary
    return phis, s


def main():
    dat = np.load(DATA / 'ch4_orbitals.npz')
    a0, N = float(dat['a0']), int(dat['N'])
    M = pwlib.cubic_cell(a0)
    G, G2 = pwlib.gvectors(M, N)
    vext = pwlib.external_potential(G, G2, M, dat['positions'], dat['charges'])
    rho = pwlib.density_from_orbitals(dat['C'], M)
    veff = pwlib.effective_potential(rho, vext, G2)
    eps = dat['eigenvalues']

    psis = [pwlib.ifft_pw(c, M) for c in dat['C'][2:5]]          # the t2 set
    phis, s = real_basis(psis)
    pwlib.check('only three independent real functions (s_4 / s_1)', s[3] / s[0],
                0.0, 1e-6)
    phis *= np.sqrt(N**3 / pwlib.cell_volume(M))                 # grid normalization

    # verification: real, orthonormal, eigenfunctions, same density
    S = np.array([[pwlib.integrate(a * b, M) for b in phis] for a in phis])
    pwlib.check('new orbitals are orthonormal', np.abs(S - np.eye(3)).max(), 0.0, 1e-10)
    for k, phi in enumerate(phis):
        c = pwlib.fft_pw(phi, M)
        r = np.linalg.norm(pwlib.apply_hamiltonian(c, veff, G2, M) - eps[2 + k] * c)
        pwlib.check('residual |Hc - eps c|, t2 orbital %i' % (k + 1), r, 0.0, 1e-5)
    rho_old = 2 * sum(np.abs(p)**2 for p in psis)
    rho_new = 2 * sum(phi**2 for phi in phis)
    pwlib.check('t2 density unchanged', np.abs(rho_new - rho_old).max(), 0.0, 1e-10)

    # optional: rotate onto p-like functions along x, y and z (Procrustes)
    r = pwlib.realspace_grid(M, N)
    d = r - dat['positions'][0]
    dist = np.linalg.norm(d, axis=-1)
    ps = [d[..., k] * np.exp(-dist) for k in range(3)]
    A = np.array([[pwlib.integrate(phi * p, M) for p in ps] for phi in phis])
    U, _, Vt = np.linalg.svd(A)
    phis_xyz = np.einsum('ij,jabc->iabc', (U @ Vt).T, phis)
    for k, label in enumerate('xyz'):
        o = pwlib.integrate(phis_xyz[k] * ps[k], M) / np.sqrt(pwlib.integrate(ps[k]**2, M))
        print('overlap of t2_%s with p_%s-like function: %.4f' % (label, label, o))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    kz = N // 2
    for k, label in enumerate('xyz'):
        sl = phis_xyz[k][:, :, kz] if label != 'z' else phis_xyz[k][:, N // 2, :]
        lim = np.abs(sl).max()
        ax[k].contourf(sl.T, levels=21, extent=(0, a0, 0, a0), cmap='PiYG',
                       vmin=-lim, vmax=lim)
        ax[k].set_title(r'$t_2$ ($%s$-like), %s-plane' % (label, 'xz' if label == 'z' else 'xy'))
        ax[k].set_aspect('equal')
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'chal5_6_t2.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
