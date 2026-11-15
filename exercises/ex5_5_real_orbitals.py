# -*- coding: utf-8 -*-
"""
Exercise 5.5 -- Real-valued orbitals and FFT upsampling            [Exercise]

Complete the TODO blocks below and run this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

DATA = Path(__file__).resolve().parent.parent / 'data'


def optimal_phase(psi):
    """TODO: the phase theta that maximizes sum (Re[psi e^{i theta}])^2.
    Derive the closed-form expression first (see the description)."""
    raise NotImplementedError


def imaginary_fraction(psi):
    """sum (Im psi)^2 / sum |psi|^2"""
    return np.sum(psi.imag**2) / np.sum(np.abs(psi)**2)


def upsample(c, factor):
    """
    TODO: place the coefficients c (numpy.fft order) on a grid that is
    `factor` times finer, with zeros for the new plane waves. Along each axis,
    split the Nyquist coefficient (index -N/2) equally over +N/2 and -N/2.
    Hint: treat one axis at a time with np.moveaxis.
    """
    raise NotImplementedError


def main():
    dat = np.load(DATA / 'ch4_orbitals.npz')
    a0, N = float(dat['a0']), int(dat['N'])
    M = pwlib.cubic_cell(a0)
    names = ['1a1', '2a1', 't2 (a)', 't2 (b)', 't2 (c)']

    # (a) TODO: for every orbital, the imaginary fraction before and after
    # multiplication by exp(i theta_opt), and after the recipe of the primer
    # (phase taken at the grid point of largest |psi|); store the orbitals
    # rotated with theta_opt in `real`
    real = []
    pwlib.check('2a1 becomes real (Im-fraction)', imaginary_fraction(real[1]), 0.0, 1e-10)

    # (b) upsampling the 2a1 orbital by a factor of four
    c = pwlib.fft_pw(real[1], M)
    psi4 = pwlib.ifft_pw(upsample(c, 4), M)
    pwlib.check('original grid points are reproduced',
                np.abs(psi4[::4, ::4, ::4] - real[1]).max(), 0.0, 1e-12)
    pwlib.check('upsampled orbital remains real', imaginary_fraction(psi4), 0.0, 1e-10)
    # TODO: compare the norm on the fine grid with that on the coarse grid,
    # and explain the difference
    # TODO: contour plots of the 2a1 orbital in the plane x = y (which
    # contains C and two H atoms), before and after upsampling
    plt.show()


if __name__ == '__main__':
    main()
