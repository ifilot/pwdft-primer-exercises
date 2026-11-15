# -*- coding: utf-8 -*-
"""
Exercise 3.4 -- The local density approximation                    [Exercise]

1. Write lda_x and lda_xc in pwlib.py (vwn_c is provided).
2. Complete the TODO blocks below and run this script.
"""

import numpy as np

import pwlib


def main():
    a0, N = 10.0, 32
    M = pwlib.cubic_cell(a0)
    r = pwlib.realspace_grid(M, N)
    R = np.array([5.0, 5.0, 5.0])
    alpha = 2.0
    rho = pwlib.gaussian_density(r, M, R, alpha)

    # (a) Slater exchange energy versus the analytical result
    ex, vx = pwlib.lda_x(rho)
    Ex = pwlib.integrate(rho * ex, M)
    # TODO: the analytical value; derive int rho^(4/3) dr for the Gaussian
    Ex_exact = None
    pwlib.check('E_x of the Gaussian density', Ex, Ex_exact, 1e-8)

    # (b) the potentials are functional derivatives of the energies
    drho = rho * np.cos(2 * np.pi * r[..., 0] / a0)   # a smooth perturbation
    eps = 1e-4
    for label, fun in (('v_x', pwlib.lda_x), ('v_xc', pwlib.lda_xc)):
        def energy(n):
            return pwlib.integrate(n * fun(n)[0], M)
        # TODO: central finite difference of energy(rho + eps * drho) in eps
        fd = None
        an = pwlib.integrate(fun(rho)[1] * drho, M)
        pwlib.check('dE/d(eps) vs integral of %s * drho' % label, fd, an, 1e-7)

    # (c) TODO: the gradient of rho from its plane-wave expansion, compared
    # with the analytical gradient -2 alpha (r - R) rho, for N = 32 and 48

    # (d) TODO: which fraction of the grid points has rho < 1e-10, and what
    # do they contribute to E_xc?


if __name__ == '__main__':
    main()
