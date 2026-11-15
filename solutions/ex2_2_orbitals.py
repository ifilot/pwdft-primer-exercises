# -*- coding: utf-8 -*-
"""
Exercise 2.2 -- One orbital, two representations                  [SOLUTION]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

DATA = Path(__file__).resolve().parent.parent / 'data'
OUTPUT = Path(__file__).resolve().parent / 'output'


def main():
    #
    # (a) the 1s orbital of hydrogen, as in the demonstration of the primer
    #
    a0, N = 10.0, 32
    M = pwlib.cubic_cell(a0)
    r = pwlib.realspace_grid(M, N)
    G, G2 = pwlib.gvectors(M, N)
    d = np.linalg.norm(r - np.array([5.0, 5.0, 5.0]), axis=-1)
    psi = np.exp(-d) / np.sqrt(np.pi)

    c = pwlib.fft_pw(psi, M)
    norm_r = pwlib.integrate(psi**2, M)
    norm_g = np.sum(np.abs(c)**2)
    ekin = pwlib.kinetic_energy(c, G2)
    print('H 1s, N = 32: <psi|psi> (real space)  = %.10f' % norm_r)
    print('              sum |c(G)|^2            = %.10f' % norm_g)
    print('              E_kin                   = %.10f (exact 0.5)' % ekin)
    pwlib.check('Parseval: both norms are identical', norm_r - norm_g, 0.0, 1e-12)
    # reference values: data behind the convergence figure of the primer
    pwlib.check('norm at N = 32 (primer, Fig. 4)', norm_r, 1.00077013, 1e-8)
    pwlib.check('E_kin at N = 32 (primer, Fig. 4)', ekin, 0.51662733, 1e-8)

    #
    # (b) the 1pi orbitals of CO (HF/STO-3G)
    #
    dat = np.load(DATA / 'co_1pi.npz')
    psi5, psi6 = dat['psi']
    a0, N = float(dat['a0']), int(dat['N'])
    M = pwlib.cubic_cell(a0)
    G, G2 = pwlib.gvectors(M, N)

    # contour plots: plane through the molecular axis (xz, y = 5) and the
    # plane perpendicular to it halfway between the nuclei (xy)
    zmid = np.mean(dat['positions'][:, 2])
    kz = int(round(zmid / a0 * N))
    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
    ext = (0, a0, 0, a0)
    for col, (name, p) in enumerate((('psi5 (pi_x)', psi5), ('psi6 (pi_y)', psi6))):
        lim = np.abs(p).max()
        ax[0, col].contourf(p[:, N // 2, :].T, levels=21, extent=ext,
                            cmap='PiYG', vmin=-lim, vmax=lim)
        ax[0, col].set_title('%s, xz-plane (y = %.1f)' % (name, a0 / 2))
        ax[0, col].set_xlabel('x [bohr]')
        ax[0, col].set_ylabel('z [bohr]')
        ax[1, col].contourf(p[:, :, kz].T, levels=21, extent=ext,
                            cmap='PiYG', vmin=-lim, vmax=lim)
        ax[1, col].set_title('%s, xy-plane (z = %.2f)' % (name, kz * a0 / N))
        ax[1, col].set_xlabel('x [bohr]')
        ax[1, col].set_ylabel('y [bohr]')
    for a in ax.ravel():
        a.set_aspect('equal')
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'ex2_2_co_1pi.png', dpi=150)

    # orthonormality in real space and in reciprocal space
    c5, c6 = pwlib.fft_pw(psi5, M), pwlib.fft_pw(psi6, M)
    S_r = np.array([[pwlib.integrate(a * b, M) for b in (psi5, psi6)]
                    for a in (psi5, psi6)])
    S_g = np.array([[np.sum(np.conj(a) * b) for b in (c5, c6)]
                    for a in (c5, c6)])
    print('overlap matrix, real space:\n', S_r)
    print('overlap matrix, reciprocal space:\n', S_g.real)
    pwlib.check('<5|5> (real space)', S_r[0, 0], 1.0, 1e-4)
    pwlib.check('<6|6> (real space)', S_r[1, 1], 1.0, 1e-4)
    pwlib.check('<5|6> (real space)', S_r[0, 1], 0.0, 1e-10)
    pwlib.check('real and reciprocal overlaps agree',
                np.abs(S_r - S_g).max(), 0.0, 1e-12)

    # kinetic energy in reciprocal space versus the exact Gaussian-basis value
    T5, T6 = pwlib.kinetic_energy(c5, G2), pwlib.kinetic_energy(c6, G2)
    print('E_kin(psi5) = %.6f, E_kin(psi6) = %.6f, exact = %.6f'
          % (T5, T6, dat['ekin_exact'][0]))
    pwlib.check('E_kin(psi5) vs exact', T5, dat['ekin_exact'][0], 1e-3)
    pwlib.check('E_kin(psi5) = E_kin(psi6)', T5 - T6, 0.0, 1e-10)

    #
    # (c) translation only changes the phases
    #
    shift = np.array([3, -5, 7])                # in grid points
    psi_t = np.roll(psi5, shift, axis=(0, 1, 2))
    c_t = pwlib.fft_pw(psi_t, M)
    dvec = shift * a0 / N
    pwlib.check('c_t(G) = c(G) exp(-i G.d)',
                np.abs(c_t - c5 * np.exp(-1j * G @ dvec)).max(), 0.0, 1e-12)
    pwlib.check('|c_t(G)| = |c(G)|', np.abs(np.abs(c_t) - np.abs(c5)).max(),
                0.0, 1e-12)
    pwlib.check('kinetic energy unchanged', pwlib.kinetic_energy(c_t, G2) - T5,
                0.0, 1e-12)
    plt.show()


if __name__ == '__main__':
    main()
