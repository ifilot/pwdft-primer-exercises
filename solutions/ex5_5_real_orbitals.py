# -*- coding: utf-8 -*-
"""
Exercise 5.5 -- Real-valued orbitals and FFT upsampling            [SOLUTION]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

DATA = Path(__file__).resolve().parent.parent / 'data'
OUTPUT = Path(__file__).resolve().parent / 'output'


def optimal_phase(psi):
    """
    Phase theta maximizing sum (Re[psi e^{i theta}])^2. Writing
    (Re z)^2 = (|z|^2 + Re z^2) / 2 shows that this is achieved when
    2 theta = -arg(sum psi^2).
    """
    return -0.5 * np.angle(np.sum(psi**2))


def imaginary_fraction(psi):
    """sum (Im psi)^2 / sum |psi|^2"""
    return np.sum(psi.imag**2) / np.sum(np.abs(psi)**2)


def upsample(c, factor):
    """
    Place the coefficients c (numpy.fft order) on a grid that is `factor`
    times finer, with zeros for all new plane waves. Along each axis, the
    Nyquist coefficient (index -N/2) is split equally over +N/2 and -N/2.
    """
    for axis in range(3):
        c = np.moveaxis(c, axis, 0)
        N = c.shape[0]
        out = np.zeros((factor * N,) + c.shape[1:], dtype=complex)
        out[:N // 2] = c[:N // 2]                      # 0 .. N/2-1
        out[-(N // 2) + 1:] = c[-(N // 2) + 1:]        # -N/2+1 .. -1
        out[N // 2] = 0.5 * c[N // 2]                  # +N/2
        out[-(N // 2)] = 0.5 * c[N // 2]               # -N/2
        c = np.moveaxis(out, 0, axis)
    return c


def main():
    dat = np.load(DATA / 'ch4_orbitals.npz')
    a0, N = float(dat['a0']), int(dat['N'])
    M = pwlib.cubic_cell(a0)
    _, G2 = pwlib.gvectors(M, N)
    names = ['1a1', '2a1', 't2 (a)', 't2 (b)', 't2 (c)']

    # (a) one phase factor per orbital
    print('orbital   Im-fraction before   after optimal phase   after max-point phase')
    real = []
    for name, c in zip(names, dat['C']):
        psi = pwlib.ifft_pw(c, M)
        psi_opt = psi * np.exp(1j * optimal_phase(psi))
        k = np.argmax(np.abs(psi))
        psi_max = psi * np.exp(-1j * np.angle(psi.flat[k]))
        print('%-8s  %.3e            %.3e             %.3e'
              % (name, imaginary_fraction(psi), imaginary_fraction(psi_opt),
                 imaginary_fraction(psi_max)))
        real.append(psi_opt)
        c_opt = pwlib.fft_pw(psi_opt, M)
        assert abs(pwlib.kinetic_energy(c_opt, G2) - pwlib.kinetic_energy(c, G2)) < 1e-10
    pwlib.check('2a1 becomes real (Im-fraction)', imaginary_fraction(real[1]), 0.0, 1e-10)

    # (b) upsampling the 2a1 orbital by a factor of four
    c = pwlib.fft_pw(real[1], M)
    c4 = upsample(c, 4)
    psi4 = pwlib.ifft_pw(c4, M)
    pwlib.check('original grid points are reproduced',
                np.abs(psi4[::4, ::4, ::4] - real[1]).max(), 0.0, 1e-12)
    # every split halves the weight of a Nyquist coefficient in the norm
    nyq = np.arange(N) == N // 2
    w = np.ones(c.shape)
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = N
        w *= np.where(nyq, 0.5, 1.0).reshape(shape)
    norm4 = pwlib.integrate(np.abs(psi4)**2, M)
    print('norm on the fine grid: %.8f (coarse grid: %.8f)'
          % (norm4, np.sum(np.abs(c)**2)))
    pwlib.check('fine-grid norm = sum |c|^2 w (Nyquist weights)', norm4,
                np.sum(w * np.abs(c)**2), 1e-12)
    pwlib.check('upsampled orbital remains real', imaginary_fraction(psi4), 0.0, 1e-10)
    nonsplit = upsample(c, 4)
    nonsplit[-16, :, :] *= 2.0  # along x: full Nyquist coefficient at -N/2 ...
    nonsplit[16, :, :] = 0.0    # ... and nothing at +N/2
    print('without the Nyquist split, Im-fraction = %.1e'
          % imaginary_fraction(pwlib.ifft_pw(nonsplit, M)))

    # plot in the plane x = y, which contains C and two H atoms
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, p, n in ((ax[0], real[1], N), (ax[1], psi4.real, 4 * N)):
        idx = np.arange(n)
        sl = np.real(p)[idx, idx, :]                 # rows: s along (1,1,0)
        s = idx * a0 / n * np.sqrt(2)
        z = idx * a0 / n
        lim = np.abs(sl).max()
        a.contourf(s, z, sl.T, levels=21, cmap='PiYG', vmin=-lim, vmax=lim)
        a.contour(s, z, sl.T, levels=[0.0], colors='k', linewidths=0.5)
        a.set_title('2a1, N = %i' % n)
        a.set_xlabel(r'$s$ along $(1,1,0)$ [bohr]')
        a.set_ylabel(r'$z$ [bohr]')
        a.set_aspect('equal')
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / 'ex5_5_upsampling.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
