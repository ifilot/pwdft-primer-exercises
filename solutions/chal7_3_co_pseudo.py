# -*- coding: utf-8 -*-
"""
Challenge 7.3 -- Carbon monoxide with pseudopotentials             [SOLUTION]

Requires PyPWDFT >= 0.3.0.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pypwdft import PWDFT

import pwlib

OUTPUT = Path(__file__).resolve().parent / 'output'
RUN_BOND_SCAN = False          # part (d): about 10 minutes

# GTH-PBE parameters of the s projectors (pypwdft/psp/pbe/C-q4 and O-q6)
PROJ = {'C': (0.30257575, 9.62248665), 'O': (0.22095592, 18.33745811)}


def main():
    L = 10.0
    ecuts = [20, 40, 60, 80, 120, 160]
    res = {}
    for ecut in ecuts:
        res[ecut] = PWDFT(pwlib.diatomic('CO', L), cutoff=ecut, xc='pbe',
                          pseudopotential='gth').run(bands=7)
        r = res[ecut]
        print('E_cut = %3d Ht  N_pw = %6d  E = %.6f Ht' % (ecut, r.basis.plane_waves,
                                                         r.energy.total))

    # (a) convergence with respect to the value at 160 Ht
    eref = res[160].energy.total
    pwlib.check('E_total at 120 Ht (primer: about -21.69 Ht)', res[120].energy.total,
                -21.69, 5e-3)
    pwlib.check('N_pw at 120 Ht (primer: about 63 000)', res[120].basis.plane_waves,
                63000, 1000)
    pwlib.check('error at 20 Ht (primer: 0.7 Ht)', res[20].energy.total - eref, 0.7, 0.05)
    pwlib.check('error at 120 Ht (primer: about 1 mHt)', res[120].energy.total - eref,
                1e-3, 1e-3)

    # (b) valence levels relative to the HOMO, and their order
    print('\nE_cut   3s-5s     4s-5s     1pi-5s   (Ht)   order correct?')
    for ecut in ecuts:
        e = res[ecut].orbitals.energies
        print('%4d   %+.4f  %+.4f  %+.4f        %s'
              % (ecut, e[0] - e[4], e[1] - e[4], e[2] - e[4], e[1] < e[2] < e[4]))

    # (c) the projectors, built with pwlib, applied to PyPWDFT's orbitals
    r = res[80]
    M = pwlib.cubic_cell(L)
    psis = pwlib.from_pypwdft(r.orbitals.real_space)     # [x, y, z] convention
    N = psis.shape[1]
    G, G2 = pwlib.gvectors(M, N)
    pos = pwlib.diatomic('CO', L).positions
    proj = {el: pwlib.gth_projector_s_g(G, G2, M, R, PROJ[el][0])
            for el, R in zip(('C', 'O'), pos)}
    labels = ['3sigma', '4sigma', '1pi', '1pi', '5sigma', '2pi*', '2pi*']
    print('\norbital   |<p_C|psi>|   |<p_O|psi>|')
    enl = 0.0
    for k, psi in enumerate(psis):
        c = pwlib.fft_pw(psi, M)
        ov = {el: np.vdot(p, c) for el, p in proj.items()}
        print('%-7s   %.2e      %.2e' % (labels[k], abs(ov['C']), abs(ov['O'])))
        if k < 5:      # doubly occupied
            enl += 2 * sum(PROJ[el][1] * abs(ov[el])**2 for el in ov)
    pwlib.check('E_nl from own projectors = PyPWDFT E_nl', enl, r.energy.nonlocal_, 1e-8)

    # (d) optional: equilibrium bond length (primer: 1.1356 to 1.1357 A)
    if RUN_BOND_SCAN:
        bonds = np.linspace(1.125, 1.145, 5)
        E = [PWDFT(pwlib.diatomic('CO', 12.0, bond=b), cutoff=120, xc='pbe',
                   pseudopotential='gth').run().energy.total for b in bonds]
        c2, c1, _ = np.polyfit(bonds, E, 2)
        print('equilibrium bond length: %.4f A' % (-c1 / (2 * c2)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(ecuts[:-1], [res[e].energy.total - eref for e in ecuts[:-1]], 'o-')
    ax.axhline(1e-3, ls=':', color='k', lw=0.8)
    ax.set_xlabel(r'$E_\mathrm{cut}$ [Ht]')
    ax.set_ylabel(r'$E(E_\mathrm{cut}) - E(160)$ [Ht]')
    ax.grid(ls='--', alpha=0.5, which='both')
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'chal7_3_co_pseudo.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
