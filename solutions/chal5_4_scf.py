# -*- coding: utf-8 -*-
"""
Challenge 5.4 -- Your own self-consistent field program              [SOLUTION]
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pwlib

OUTPUT = Path(__file__).resolve().parent / 'output'
RUN_MIXING_STUDY = True

CH4 = (np.array([[5.0000, 5.0000, 5.0000],
                 [6.1958, 6.1958, 6.1958],
                 [3.8042, 3.8042, 6.1958],
                 [3.8042, 6.1958, 3.8042],
                 [6.1958, 3.8042, 3.8042]]),
       np.array([6.0, 1.0, 1.0, 1.0, 1.0]))
H2 = (np.array([[5.0, 5.0, 4.3], [5.0, 5.0, 5.7]]), np.array([1.0, 1.0]))


def scf(a0, N, positions, charges, beta=0.5, etol=1e-7, rhotol=1e-6,
        maxiter=100, verbose=True):
    """
    Minimal Kohn-Sham SCF cycle: Gamma-point, bare Coulomb nuclei, LDA,
    all plane waves on the grid as basis, linear density mixing.
    """
    M = pwlib.cubic_cell(a0)
    G, G2 = pwlib.gvectors(M, N)
    nelec = int(round(np.sum(charges)))
    nocc = nelec // 2
    vext = pwlib.external_potential(G, G2, M, positions, charges)
    eewald = pwlib.ewald_energy(M, positions, charges)

    rho = np.full(G2.shape, nelec / pwlib.cell_volume(M))   # homogeneous guess
    C, E_old = None, 0.0
    history = []
    for it in range(1, maxiter + 1):
        veff = pwlib.effective_potential(rho, vext, G2)
        eps, C = pwlib.lowest_eigenstates(veff, G2, M, nocc, C0=C)
        rho_new = pwlib.density_from_orbitals(C, M)
        E = pwlib.energy_components(C, M, G2, vext, eewald)
        drho = np.sqrt(np.mean((rho_new - rho)**2))
        dE = E['total'] - E_old
        history.append((E['total'], drho))
        if verbose:
            print('%3i  E = %+.10f  dE = %+.2e  rms(drho) = %.2e'
                  % (it, E['total'], dE, drho))
        if abs(dE) < etol and drho < rhotol:
            return {'energy': E, 'eigenvalues': eps, 'C': C, 'rho': rho_new,
                    'iterations': it, 'history': np.array(history)}
        rho = beta * rho_new + (1.0 - beta) * rho
        E_old = E['total']
    raise RuntimeError('SCF did not converge in %i iterations' % maxiter)


def main():
    # (a) H2
    t0 = time.perf_counter()
    res = scf(10.0, 32, *H2)
    print('H2: E = %.6f Ht, eps_1 = %.6f Ht, %i iterations, %.1f s\n'
          % (res['energy']['total'], res['eigenvalues'][0], res['iterations'],
             time.perf_counter() - t0))
    pwlib.check('H2: total energy', res['energy']['total'], -1.1373616, 1e-6)

    # (b) CH4: compare with the reference set of orbitals of Exercise 5.1
    t0 = time.perf_counter()
    res = scf(10.0, 32, *CH4)
    print('CH4: %i iterations, %.1f s' % (res['iterations'], time.perf_counter() - t0))
    for k, v in res['energy'].items():
        print('    %-8s %+.6f' % (k, v))
    print('    eigenvalues:', np.round(res['eigenvalues'], 5))
    ref = np.load(Path(__file__).resolve().parent.parent / 'data' / 'ch4_orbitals.npz')
    pwlib.check('CH4: total energy vs data file',
                res['energy']['total'], float(ref['total_energy']), 1e-6)
    pwlib.check('CH4: eigenvalues vs data file',
                np.abs(res['eigenvalues'] - ref['eigenvalues']).max(), 0.0, 1e-5)

    # (c) mixing: CH4 versus beta, and H2 in cells of increasing size
    if RUN_MIXING_STUDY:
        betas = [0.2, 0.4, 0.6, 0.8, 1.0]
        fig, ax = plt.subplots(figsize=(6, 4))
        cases = [('CH4, a0 = 10', 10.0, 32, CH4)] + [
            ('H2, a0 = %g' % a0, a0, N, H2) for a0, N in ((10.0, 32), (15.0, 48), (20.0, 64))]
        for label, a0, N, (pos, chg) in cases:
            its = []
            for beta in betas:
                try:
                    its.append(scf(a0, N, pos + (a0 - 10.0) / 2, chg, beta=beta,
                                   maxiter=60, verbose=False)['iterations'])
                except RuntimeError:
                    its.append(np.nan)            # not converged in 60 iterations
            print('%-14s iterations for beta = %s: %s' % (label, betas, its))
            ax.plot(betas, its, 'o-', label=label)
        ax.set_xlabel(r'mixing parameter $\beta$')
        ax.set_ylabel('SCF iterations (max. 60)')
        ax.legend()
        ax.grid(ls='--', alpha=0.5)
        fig.tight_layout()
        OUTPUT.mkdir(exist_ok=True)
        fig.savefig(OUTPUT / 'chal5_4_mixing.png', dpi=150)
        plt.show()


if __name__ == '__main__':
    main()
