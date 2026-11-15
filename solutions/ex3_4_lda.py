# -*- coding: utf-8 -*-
"""
Exercise 3.4 -- The local density approximation                   [SOLUTION]
"""

import numpy as np

import pwlib


def main():
    a0, N = 10.0, 32
    M = pwlib.cubic_cell(a0)
    r = pwlib.realspace_grid(M, N)
    G, G2 = pwlib.gvectors(M, N)
    R = np.array([5.0, 5.0, 5.0])
    alpha = 2.0
    rho = pwlib.gaussian_density(r, M, R, alpha)

    # (a) Slater exchange energy versus the analytical result
    ex, vx = pwlib.lda_x(rho)
    Ex = pwlib.integrate(rho * ex, M)
    Ex_exact = (-0.75 * (3 / np.pi)**(1 / 3) * (alpha / np.pi)**2
                * (3 * np.pi / (4 * alpha))**1.5)
    pwlib.check('E_x of the Gaussian density', Ex, Ex_exact, 1e-8)
    exc, vxc = pwlib.lda_xc(rho)
    Exc = pwlib.integrate(rho * exc, M)
    print('E_x = %.8f   E_c = %.8f   E_xc = %.8f' % (Ex, Exc - Ex, Exc))

    # (b) the potentials are functional derivatives of the energies
    # a smooth perturbation that is everywhere small compared with rho itself
    drho = rho * np.cos(2 * np.pi * r[..., 0] / a0)
    eps = 1e-4
    for label, fun in (('v_x', pwlib.lda_x), ('v_xc', pwlib.lda_xc)):
        def energy(n):
            return pwlib.integrate(n * fun(n)[0], M)
        fd = (energy(rho + eps * drho) - energy(rho - eps * drho)) / (2 * eps)
        an = pwlib.integrate(fun(rho)[1] * drho, M)
        pwlib.check('dE/d(eps) vs integral of %s * drho' % label, fd, an, 1e-7)

    # optional: a broad perturbation that does not vanish where rho does; in
    # the vacuum, eps * drho exceeds rho, so E(rho + eps drho) is no longer
    # linear in eps there and the finite difference deviates
    broad = np.cos(2 * np.pi * r[..., 0] / a0)
    for label, fun in (('v_x', pwlib.lda_x), ('v_xc', pwlib.lda_xc)):
        def energy(n):
            return pwlib.integrate(n * fun(n)[0], M)
        fd = (energy(rho + eps * broad) - energy(rho - eps * broad)) / (2 * eps)
        an = pwlib.integrate(fun(rho)[1] * broad, M)
        print('broad perturbation, %-4s: finite difference %+.6f, integral %+.6f '
              '(relative deviation %.0f %%)' % (label, fd, an, 100 * abs(fd / an - 1)))

    # (c) the gradient of the density from its plane-wave expansion
    for Ng, tol in ((32, 1e-5), (48, 1e-10)):
        Mg = pwlib.cubic_cell(a0)
        rg = pwlib.realspace_grid(Mg, Ng)
        Gg, _ = pwlib.gvectors(Mg, Ng)
        rhon = pwlib.gaussian_density(rg, Mg, R, alpha)
        rhog = pwlib.fft_pw(rhon, Mg)
        grad = np.stack([pwlib.ifft_pw(1j * Gg[..., k] * rhog, Mg).real
                         for k in range(3)], axis=-1)
        d = pwlib.minimum_image(rg, R, Mg)
        grad_exact = -2 * alpha * d * rhon[..., None]
        pwlib.check('max |grad rho (FFT) - exact|, N = %i' % Ng,
                    np.abs(grad - grad_exact).max(), 0.0, tol)

    # (d) the vacuum
    small = rho < 1e-10
    print('fraction of grid points with rho < 1e-10: %.3f' % small.mean())
    print('their contribution to E_xc: %.2e Ht'
          % pwlib.integrate(np.where(small, rho * exc, 0.0), M))


if __name__ == '__main__':
    main()
