# -*- coding: utf-8 -*-
"""
Generate the reference Kohn-Sham orbitals of CH4 used in Exercises 5.1, 5.2,
5.5 and Challenge 5.6, by running the SCF program of the solution to Challenge 5.4.

Settings: cubic cell a0 = 10 bohr, N = 32, Gamma-point, bare Coulomb nuclei,
LDA (Slater exchange + VWN correlation, variant pwlib.VWN_DEFAULT), all plane
waves on the grid as basis. The eigensolver returns each orbital with an
arbitrary phase, and the degenerate t2 orbitals as an arbitrary (complex)
mixture; this is deliberate (see Exercise 5.5 and Challenge 5.6).

This script is not needed by students.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'solutions'))

import pwlib                    # noqa: E402
import chal5_4_scf as chal      # noqa: E402

OUT = ROOT / 'data' / 'ch4_orbitals.npz'


def main():
    positions, charges = chal.CH4
    res = chal.scf(10.0, 32, positions, charges, etol=1e-9, rhotol=1e-7)

    # give every orbital a random global phase so that students cannot rely
    # on the (implementation-dependent) phase returned by ARPACK
    rng = np.random.default_rng(2026)
    C = res['C'] * np.exp(2j * np.pi * rng.random(len(res['C'])))[:, None, None, None]

    np.savez_compressed(OUT, C=C, eigenvalues=res['eigenvalues'], a0=10.0, N=32,
                        positions=positions, charges=charges,
                        total_energy=res['energy']['total'],
                        functional='Slater + %s' % pwlib.VWN_DEFAULT.upper())
    print('Saved', OUT)
    for k, v in res['energy'].items():
        print('  %-8s %+.8f' % (k, v))
    print('  eigenvalues', res['eigenvalues'])


if __name__ == '__main__':
    main()
