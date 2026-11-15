# -*- coding: utf-8 -*-
"""
Exercise 6.1 -- A complete calculation with PyPWDFT                [Exercise]

Requires PyPWDFT >= 0.3.0 (pip install pypwdft). Complete the TODO blocks and
run this script.
"""

from pathlib import Path

import numpy as np
from pypwdft import PWDFT

import pwlib

OUTPUT = Path(__file__).resolve().parent / 'output'
E_REF_H2 = -1.16673          # Gaussian basis-set limit, PBE (primer, Section 6)


def main():
    L = 15.0
    results = {}        # the calculations for parts (a) to (c)
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

    # (b) the number of plane waves
    for ecut, r in results.items():
        # TODO: estimate the number of plane waves inside the sphere
        # 1/2 |G|^2 <= E_cut from the volume per G-vector
        estimate = None
        pwlib.check('N_pw estimate, E_cut = %d' % ecut,
                    r.basis.plane_waves / estimate, 1.0, 0.01)

    # (c) TODO: tabulate E_total, its error and eps_1 for the three cut-offs

    # (c) the occupied orbital, made real-valued by PyPWDFT
    OUTPUT.mkdir(exist_ok=True)
    results[80].plot_orbitals(plane='xz', save=str(OUTPUT / 'ex6_1_h2_orbital.png'))


if __name__ == '__main__':
    main()
