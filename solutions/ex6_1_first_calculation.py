# -*- coding: utf-8 -*-
"""
Exercise 6.1 -- A complete calculation with PyPWDFT                [SOLUTION]

Requires PyPWDFT >= 0.3.0 (pip install pypwdft).
"""

from pathlib import Path

import numpy as np
from pypwdft import PWDFT

import pwlib

OUTPUT = Path(__file__).resolve().parent / 'output'
E_REF_H2 = -1.16673          # Gaussian basis-set limit, PBE (primer, Section 6)


def main():
    L = 15.0
    results = {}
    for ecut in (20, 40, 80):
        calc = PWDFT(pwlib.diatomic('H2', L), cutoff=ecut, xc='pbe',
                     pseudopotential='all-electron')
        results[ecut] = calc.run()

    # (a) the calculation at 20 Ht
    res = results[20]
    print(res)
    print(res.basis)
    for name in ('kinetic', 'electron_ion', 'hartree', 'xc', 'ion_ion', 'total'):
        print('  %-13s %+.6f Ht' % (name, getattr(res.energy, name)))
    pwlib.check('error at 20 Ht (primer: 15 mHt)', res.energy.total - E_REF_H2,
                0.015, 1e-3)

    # (b) number of plane waves: the volume of the sphere |G| <= sqrt(2 E_cut),
    # divided by the volume (2 pi)^3 / Omega per G-vector
    for ecut, r in results.items():
        estimate = L**3 / (2 * np.pi)**3 * 4 * np.pi / 3 * (2 * ecut)**1.5
        print('E_cut = %3d Ht: N_pw = %6d (estimate %8.0f), density grid %d^3'
              % (ecut, r.basis.plane_waves, estimate, r.basis.density_grid))
        pwlib.check('N_pw estimate, E_cut = %d' % ecut,
                    r.basis.plane_waves / estimate, 1.0, 0.01)

    # (c) convergence with the cut-off energy
    print('\nE_cut   E_total       error [mHt]   eps_1 [Ht]')
    for ecut, r in results.items():
        print('%4d   %.6f   %8.2f     %.5f' % (ecut, r.energy.total,
                                              1000 * (r.energy.total - E_REF_H2),
                                              r.orbitals.energies[0]))

    # (c) the occupied orbital, made real-valued by PyPWDFT
    OUTPUT.mkdir(exist_ok=True)
    results[80].plot_orbitals(plane='xz', save=str(OUTPUT / 'ex6_1_h2_orbital.png'))


if __name__ == '__main__':
    main()
