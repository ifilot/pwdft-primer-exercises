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

    # diagonalize once more in the potential of the FINAL density, so that the
    # stored orbitals and eigenvalues belong to one and the same Hamiltonian
    M = pwlib.cubic_cell(10.0)
    G, G2 = pwlib.gvectors(M, 32)
    vext = pwlib.external_potential(G, G2, M, positions, charges)
    veff = pwlib.effective_potential(res['rho'], vext, G2)
    eps, C0 = pwlib.lowest_eigenstates(veff, G2, M, len(res['C']), C0=res['C'],
                                       tol=1e-10, maxiter=1000)
    res['eigenvalues'], res['C'] = eps, C0
    res['energy'] = pwlib.energy_components(C0, M, G2, vext,
                                            pwlib.ewald_energy(M, positions, charges))
    rho = pwlib.density_from_orbitals(C0, M)
    veff = pwlib.effective_potential(rho, vext, G2)
    resid = max(np.linalg.norm(pwlib.apply_hamiltonian(c, veff, G2, M) - e * c)
                for c, e in zip(C0, eps))
    print('max residual |H[rho(C)] c - eps c| = %.1e' % resid)

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
