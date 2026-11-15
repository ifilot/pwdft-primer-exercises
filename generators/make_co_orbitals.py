# -*- coding: utf-8 -*-
"""
Generate the doubly degenerate 1pi molecular orbitals of CO (HF/STO-3G) on a
real-space grid, as used in Exercise 2.2.

This script is NOT needed by students; it documents where data/co_1pi.npz
comes from. It requires PyQInt (pip install pyqint).

Conventions (identical to pwlib):
  * cubic cell of edge a0 = 10 bohr, N = 64 points per direction
  * array index [i, j, k] <-> r = (i, j, k) * a0 / N, i.e. x is the FIRST
    (slowest) index and z the LAST (fastest) index
  * the molecule lies along z; the C-O midpoint region is near the cell centre
"""

from pathlib import Path

import numpy as np
from pyqint import HF, MoleculeBuilder, PyQInt

A0 = 10.0
N = 64
OUT = Path(__file__).resolve().parent.parent / 'data' / 'co_1pi.npz'


def main():
    mol = MoleculeBuilder().from_name('CO')
    res = HF(mol, 'sto3g').rhf()
    orbe = res['orbe']
    C = res['orbc']
    cgfs = res['cgfs']

    # the 1pi pair are orbitals 5 and 6 (indices 4 and 5); verify degeneracy
    assert abs(orbe[4] - orbe[5]) < 1e-8, 'expected a degenerate pair'

    # rotate within the degenerate pair such that the first orbital is pi_x
    # (no p_y character) and the second pi_y (no p_x character)
    py = [k for k, c in enumerate(cgfs)
          if (c.gtos[0].l, c.gtos[0].m, c.gtos[0].n) == (0, 1, 0)]
    v4, v5 = C[:, 4], C[:, 5]
    theta = np.arctan2(-v4[py[0]], v5[py[0]])
    pix = np.cos(theta) * v4 + np.sin(theta) * v5
    piy = -np.sin(theta) * v4 + np.cos(theta) * v5
    assert np.allclose(pix[py], 0.0, atol=1e-10)

    # exact kinetic energies in the Gaussian basis
    T = res['kinetic']
    ekin = np.array([pix @ T @ pix, piy @ T @ piy])

    # sample on the periodic grid; the molecule is shifted by +a0/2 in every
    # direction so that its atoms lie inside [0, a0)
    c1d = np.arange(N) * A0 / N
    x, y, z = np.meshgrid(c1d, c1d, c1d, indexing='ij')
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3) - A0 / 2
    integrator = PyQInt()
    psi = np.array([
        integrator.plot_wavefunction(pts, v, cgfs).reshape(N, N, N)
        for v in (pix, piy)
    ])

    positions = np.array([n[0] for n in res['nuclei']]) + A0 / 2
    charges = np.array([n[1] for n in res['nuclei']], dtype=float)

    np.savez_compressed(OUT, psi=psi, a0=A0, N=N, ekin_exact=ekin,
                        orbital_energy=orbe[4], positions=positions,
                        charges=charges)
    print('Saved', OUT)
    print('positions (bohr):\n', positions)
    print('exact kinetic energies (Ht):', ekin)


if __name__ == '__main__':
    main()
