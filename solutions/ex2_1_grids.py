# -*- coding: utf-8 -*-
"""
Exercise 2.1 -- Grids, plane waves and the FFT convention       [SOLUTION]
"""

import numpy as np

import pwlib


def main():
    a0, N = 10.0, 32
    M = pwlib.cubic_cell(a0)

    # (a) the two grids
    r = pwlib.realspace_grid(M, N)
    G, G2 = pwlib.gvectors(M, N)
    pwlib.check('grid spacing r[1,0,0] - r[0,0,0]', r[1, 0, 0, 0], a0 / N, 1e-12)
    pwlib.check('last sampling point r[N-1,0,0]', r[-1, 0, 0, 0],
                a0 * (N - 1) / N, 1e-12)
    pwlib.check('G[0,0,0] is the zero vector', np.abs(G[0, 0, 0]).max(), 0.0, 1e-12)
    gmin = np.sqrt(G2[G2 > 0].min())
    pwlib.check('shortest non-zero |G| = 2 pi / a0', gmin, 2 * np.pi / a0, 1e-12)
    span = G[..., 0].max() - G[..., 0].min()
    pwlib.check('extent of G_x: 2 pi (N - 1) / a0', span,
                2 * np.pi * (N - 1) / a0, 1e-12)

    # reproduce the setting of the sampling figure: a0 = 1, N = 8
    Gs, _ = pwlib.gvectors(pwlib.cubic_cell(1.0), 8)
    m = np.unique(np.rint(Gs[..., 0] / (2 * np.pi)))
    print('G_x in units of 2 pi / a0 for a0 = 1, N = 8:', m)

    # (b) a single plane wave
    m0 = np.array([1, -2, 3])
    G0 = m0 * 2 * np.pi / a0
    phi = np.exp(1j * r @ G0) / np.sqrt(pwlib.cell_volume(M))
    c = pwlib.fft_pw(phi, M)
    idx = np.unravel_index(np.argmax(np.abs(c)), c.shape)
    print('largest coefficient at index', idx, 'with G =', G[idx])
    pwlib.check('that G equals G0', np.abs(G[idx] - G0).max(), 0.0, 1e-12)
    pwlib.check('its coefficient equals 1', c[idx].real, 1.0, 1e-12)
    pwlib.check('all other coefficients vanish',
                np.abs(np.delete(c.ravel(), np.ravel_multi_index(idx, c.shape))).max(),
                0.0, 1e-12)
    pwlib.check('kinetic energy = |G0|^2 / 2', pwlib.kinetic_energy(c, G2),
                0.5 * G0 @ G0, 1e-12)

    # (c) round trip
    f = np.random.default_rng(1).normal(size=(N, N, N))
    pwlib.check('round trip IFFT_pw[FFT_pw[f]] = f',
                np.abs(pwlib.ifft_pw(pwlib.fft_pw(f, M), M) - f).max(), 0.0, 1e-12)

    # (d) optional: a plane wave that does not fit on the grid
    m1 = np.array([N // 2 + 3, 0, 0])
    phi = np.exp(1j * r @ (m1 * 2 * np.pi / a0)) / np.sqrt(pwlib.cell_volume(M))
    c = pwlib.fft_pw(phi, M)
    idx = np.unravel_index(np.argmax(np.abs(c)), c.shape)
    print('plane wave with m = %s appears at m = %s (aliasing)'
          % (m1, np.rint(G[idx] * a0 / (2 * np.pi)).astype(int)))


if __name__ == '__main__':
    main()
