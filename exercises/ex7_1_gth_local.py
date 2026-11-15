# -*- coding: utf-8 -*-
"""
Exercise 7.1 -- The local part of a pseudopotential                [Exercise]

1. Write gth_local_g and gth_external_potential in pwlib.py (gth_local_r and
   the oxygen parameters GTH_O are provided).
2. Complete the TODO blocks below and run this script.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

import pwlib

XI = 2.837297


def main():
    p = pwlib.GTH_O
    Z, rloc, C = p['Zion'], p['rloc'], p['C']

    # (a) the G = 0 constant and the value at the nucleus
    alpha = pwlib.gth_local_g(np.array([0.0]), Z, rloc, C)[0]
    pwlib.check('alpha (primer, appendix)', alpha, 0.134, 5e-4)
    # TODO: nu_loc(0) from your gth_local_g, using
    #   nu(0) = (2 pi)^-3 int nu(G) d^3G = 1/(2 pi^2) int_0^inf nu(G) G^2 dG
    v0 = None
    pwlib.check('nu_loc(0) from nu_loc(G) (primer: about -36 Ht)', v0, -36.24, 0.01)

    # (b) TODO: plot -nu_loc(G) against 1/2 G^2 on log-log axes together with
    # 4 pi Z_ion / G^2 and 4 pi Z / G^2 (Z = 8), and find where the ratio
    # nu_loc(G) / (-4 pi Z_ion / G^2) drops below 0.5, 0.1 and 0.01

    # (b') TODO: check gth_local_g against a numerical radial Fourier
    # transform of the short-range part nu_loc(r) + Z_ion / r,
    #   4 pi int_0^inf [nu_loc(r) + Z_ion/r] sin(G r) / (G r) r^2 dr,
    # which should equal nu_loc(G) + 4 pi Z_ion / G^2 (use scipy's quad)

    # (c) the potential on the grid of a periodic cell
    for a0, N in ((10.0, 128), (15.0, 144), (20.0, 192)):
        M = pwlib.cubic_cell(a0)
        G, G2 = pwlib.gvectors(M, N)
        R = np.full(3, a0 / 2)
        v = pwlib.gth_external_potential(G, G2, M, [R], [p])
        shift = v[N // 2, N // 2, N // 2] - pwlib.gth_local_r(0.0, Z, rloc, C)
        # TODO: your prediction for the shift (see the description)
        pred = None
        pwlib.check('offset at the nucleus, a0 = %g' % a0, shift, pred, 1e-5)
    plt.show()


if __name__ == '__main__':
    main()
