# -*- coding: utf-8 -*-
"""
Exercise 4.2 -- Why the background charge does no harm            [SOLUTION]
"""

import numpy as np

import pwlib

XI = 2.837297
INCLUDE_L40 = False      # L = 40 needs N = 192 (about 2 GB of memory)


def electrostatics(L, N):
    """E_H, E_nuc and E_Ewald of a Gaussian electron centred on a proton."""
    M = pwlib.cubic_cell(L)
    r = pwlib.realspace_grid(M, N)
    G, G2 = pwlib.gvectors(M, N)
    R = np.full(3, L / 2)
    rho = pwlib.gaussian_density(r, M, R, alpha=2.0)
    vh = pwlib.hartree_potential(rho, G2)
    vext = pwlib.external_potential(G, G2, M, [R], [1.0])
    eh = 0.5 * pwlib.integrate(vh * rho, M)
    enuc = pwlib.integrate(vext * rho, M)
    eewald = pwlib.ewald_energy(M, [R], [1.0])
    return eh, enuc, eewald


def main():
    table = {10: (0.4239, -1.3136, -0.1419), 20: (0.4935, -1.4541, -0.0709),
             40: (0.5287, -1.5249, -0.0355)}
    iso = (1 / np.sqrt(np.pi), -2 * np.sqrt(2 / np.pi), 0.0)
    print('  L      E_H       E_nuc     E_Ewald     Sum')
    cells = [(10, 48), (20, 96)] + ([(40, 192)] if INCLUDE_L40 else [])
    for L, N in cells:
        e = electrostatics(L, N)
        print('%4i  %8.4f  %8.4f  %8.4f  %9.4f' % (L, *e, sum(e)))
        for name, val, ref in zip(('E_H', 'E_nuc', 'E_Ewald'), e, table[L]):
            pwlib.check('%s, L = %i (primer, table in Section 4)' % (name, L), val, ref, 5e-5)
        pwlib.check('E_Ewald = -xi / (2L), L = %i' % L, e[2], -XI / (2 * L), 1e-6)
        pwlib.check('sum = isolated value, L = %i' % L, sum(e), sum(iso), 1e-6)
    print('isolated: %8.4f  %8.4f  %8.4f  %9.4f' % (*iso, sum(iso)))


if __name__ == '__main__':
    main()
