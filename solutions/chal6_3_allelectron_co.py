# -*- coding: utf-8 -*-
"""
Challenge 6.3 -- The price of a nucleus: all-electron CO           [SOLUTION]

Requires PyPWDFT >= 0.3.0. Runtime: a few minutes (200 Ht dominates).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pypwdft import PWDFT

import pwlib

OUTPUT = Path(__file__).resolve().parent / 'output'
E_REF = {'H2': -1.16673, 'CO': -113.2424}     # primer, Section 6
CELL = {'H2': 15.0, 'CO': 10.0}
ECUTS = {'H2': [20, 40, 80, 160], 'CO': [50, 100, 200]}


def main():
    data = {}
    for name in ('H2', 'CO'):
        rows = []
        for ecut in ECUTS[name]:
            res = PWDFT(pwlib.diatomic(name, CELL[name]), cutoff=ecut, xc='pbe',
                        pseudopotential='all-electron').run(bands=9 if name == 'CO' else 1)
            rows.append((ecut, res.energy.total, res.basis.plane_waves,
                         res.orbitals.energies))
            print('%s  E_cut = %4d Ht  N_pw = %7d  E = %.6f  error = %.4f Ht'
                  % (name, ecut, rows[-1][2], rows[-1][1], rows[-1][1] - E_REF[name]))
        data[name] = rows

    # (a) reproduce numbers of the primer
    co = {r[0]: r for r in data['CO']}
    h2 = {r[0]: r for r in data['H2']}
    pwlib.check('H2: error at 20 Ht (primer: 15 mHt)', h2[20][1] - E_REF['H2'], 0.015, 1e-3)
    pwlib.check('H2: error at 160 Ht below 1 mHt', min(h2[160][1] - E_REF['H2'], 1e-3) - 1e-3,
                0.0, 1e-3)
    pwlib.check('CO: error at 50 Ht (primer: almost 24 Ht)', co[50][1] - E_REF['CO'], 23.7, 0.3)
    pwlib.check('CO: O 1s level at 50 Ht (primer: -12.5 Ht)', co[50][3][0], -12.5, 0.1)

    # (b) the valence levels relative to the HOMO. The levels must be
    # identified by symmetry, not by their position in the sorted list: the
    # doubly degenerate pair is 1pi, the other three valence levels are, from
    # low to high, 3sigma, 4sigma and 5sigma (orbitals 1 and 2 are the 1s cores)
    print('\nE_cut   3sigma    4sigma    1pi    (relative to 5sigma)   4sigma below 1pi?')
    order = {}
    for ecut in ECUTS['CO']:
        e = co[ecut][3][2:7]
        k = np.argmin(np.abs(np.diff(e)))           # the degenerate pair
        pi = 0.5 * (e[k] + e[k + 1])
        s3, s4, s5 = np.delete(e, [k, k + 1])
        order[ecut] = s4 < pi
        print('%4d   %+.4f   %+.4f   %+.4f              %s'
              % (ecut, s3 - s5, s4 - s5, pi - s5, order[ecut]))
    pwlib.check('1pi below 4sigma at 50 and 100 Ht, correct order at 200 Ht',
                float(order == {50: False, 100: False, 200: True}), 1.0, 0)
    print('HOMO-LUMO gap at 200 Ht: %.4f Ht' % (co[200][3][7] - co[200][3][6]))

    # (c) how fast does the error decay? fit error = A * E_cut^-p
    for name in ('H2', 'CO'):
        ec = np.array([r[0] for r in data[name]], dtype=float)
        err = np.array([r[1] for r in data[name]]) - E_REF[name]
        p, logA = np.polyfit(np.log(ec), np.log(err), 1)
        need = np.exp((logA - np.log(1e-3)) / -p)
        print('%s: error ~ E_cut^%.2f; 1 mHt would require E_cut ~ %.1e Ht' % (name, p, need))

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ('H2', 'CO'):
        ec = [r[0] for r in data[name]]
        ax.loglog(ec, [r[1] - E_REF[name] for r in data[name]], 'o-', label=name)
    ax.axhline(1e-3, ls=':', color='k', lw=0.8)
    ax.set_xlabel(r'$E_\mathrm{cut}$ [Ht]')
    ax.set_ylabel('error of the total energy [Ht]')
    ax.legend()
    ax.grid(ls='--', alpha=0.5, which='both')
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'chal6_3_allelectron.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
