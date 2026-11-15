# -*- coding: utf-8 -*-
"""
Exercise 7.2 -- A projector that only sees s electrons             [Exercise]

1. Write gth_projector_s_g in pwlib.py.
2. Complete the TODO blocks below and run this script.
"""

import numpy as np
from scipy.special import gamma

import pwlib


def main():
    p = pwlib.GTH_O
    rs, hs = p['rs'], p['hs']
    a0, N = 10.0, 64
    M = pwlib.cubic_cell(a0)
    r = pwlib.realspace_grid(M, N)
    G, G2 = pwlib.gvectors(M, N)
    RO = np.array([5.0, 5.0, 5.0])
    RC = RO - np.array([0.0, 0.0, 1.1352 * pwlib.ANGSTROM])   # carbon of CO

    # (a) the projector in reciprocal and in real space
    pc = pwlib.gth_projector_s_g(G, G2, M, RO, rs)
    pwlib.check('projector is normalized: sum |p(G)|^2', np.sum(np.abs(pc)**2), 1.0, 1e-8)
    pr = pwlib.ifft_pw(pc, M)
    d = np.linalg.norm(pwlib.minimum_image(r, RO, M), axis=-1)
    # TODO: the real-space GTH projector p^0_1(r) of Section 7 times Y_00 = 1/sqrt(4 pi)
    exact = None
    pwlib.check('real-space projector vs the GTH projector formula (Section 7)',
                np.abs(pr - exact).max() / exact.max(), 0.0, 1e-3)

    # (b) TODO: build normalized test functions (Gaussians of exponent 1):
    # s-type on O, p_z- and p_x-type on O, p_z- and p_x-type on C. For each,
    # compute <p|psi> from the plane-wave coefficients and h |<p|psi>|^2.

    # (c) TODO: verify for one test function that <p|psi> computed in real
    # space gives the same result


if __name__ == '__main__':
    main()
