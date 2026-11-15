# -*- coding: utf-8 -*-
"""
Exercise 7.2 -- A projector that only sees s electrons             [SOLUTION]
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
    RC = RO - np.array([0.0, 0.0, 1.1352 * 1.8897259886])   # carbon of CO

    # (a) the projector in reciprocal and in real space
    pc = pwlib.gth_projector_s_g(G, G2, M, RO, rs)
    pwlib.check('projector is normalized: sum |p(G)|^2', np.sum(np.abs(pc)**2), 1.0, 1e-8)
    pr = pwlib.ifft_pw(pc, M)
    d = np.linalg.norm(pwlib.minimum_image(r, RO, M), axis=-1)
    exact = (np.sqrt(2) * np.exp(-0.5 * d**2 / rs**2)
             / (rs**1.5 * np.sqrt(gamma(1.5)))) / np.sqrt(4 * np.pi)
    pwlib.check('real-space projector vs eq. (gth_projector)',
                np.abs(pr - exact).max() / exact.max(), 0.0, 1e-3)
    pwlib.check('projector is real in real space', np.abs(pr.imag).max(), 0.0, 1e-10)

    # (b) the angular filter: overlaps with normalized test functions
    def normalized(f):
        return f / np.sqrt(pwlib.integrate(np.abs(f)**2, M))

    dO = pwlib.minimum_image(r, RO, M)
    dC = pwlib.minimum_image(r, RC, M)
    tests = {
        's-type on O':  normalized(np.exp(-np.sum(dO**2, axis=-1))),
        'p_z-type on O': normalized(dO[..., 2] * np.exp(-np.sum(dO**2, axis=-1))),
        'p_x-type on O': normalized(dO[..., 0] * np.exp(-np.sum(dO**2, axis=-1))),
        'p_z-type on C': normalized(dC[..., 2] * np.exp(-np.sum(dC**2, axis=-1))),
        'p_x-type on C': normalized(dC[..., 0] * np.exp(-np.sum(dC**2, axis=-1))),
    }
    print('\ntest function     <p|psi>      h |<p|psi>|^2 [Ht]')
    ov = {}
    for name, f in tests.items():
        ov[name] = np.vdot(pc, pwlib.fft_pw(f, M))
        print('%-15s  %+.6f    %.6f' % (name, ov[name].real, hs * abs(ov[name])**2))
    pwlib.check('p_z on O is invisible to the projector', abs(ov['p_z-type on O']), 0.0, 1e-12)
    pwlib.check('p_x on C is invisible as well', abs(ov['p_x-type on C']), 0.0, 1e-12)
    print('p_z on C is seen: |<p|psi>| = %.4f' % abs(ov['p_z-type on C']))

    # (c) the overlap is the same in real space
    f = tests['s-type on O']
    pwlib.check('<p|psi>: reciprocal = real space',
                abs(ov['s-type on O'] - pwlib.integrate(np.conj(pr) * f, M)), 0.0, 1e-10)


if __name__ == '__main__':
    main()
