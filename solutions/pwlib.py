# -*- coding: utf-8 -*-
"""
pwlib -- a minimal plane-wave density functional theory toolkit.

You build this module yourself while working through the exercises of the
primer "Plane-wave density functional theory". Every function is tagged with
the exercise in which it is written. This is the REFERENCE version.

Conventions used throughout
---------------------------
* Atomic units (bohr, Hartree).
* The unit cell is a 3x3 matrix M whose ROWS are the lattice vectors a1, a2, a3.
* A scalar field on an N1 x N2 x N3 grid is a NumPy array f[i, j, k] holding
  the value at r = (i/N1) a1 + (j/N2) a2 + (k/N3) a3. For a cubic cell, x is
  thus the first index and z the last.
* Plane-wave coefficients are stored on the same array shape, in the order
  used by numpy.fft (G = 0 first, then positive, then negative frequencies).
* fft_pw / ifft_pw follow the expansion of the primer,
      u(r) = 1/sqrt(Omega) sum_G u~(G) exp(i G.r),
  i.e. they differ from numpy.fft by the constant C_t = sqrt(Omega) / N.
"""

import os
import warnings

# Limit the number of BLAS/OpenMP threads. The grids in these exercises are
# small, and many threads mostly compete with each other (in particular inside
# ARPACK), which can make a calculation ten times SLOWER. Override with the
# environment variable PWLIB_THREADS.
MAX_THREADS = int(os.environ.get('PWLIB_THREADS', 4))
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, str(MAX_THREADS))   # effective if numpy is not yet loaded

import numpy as np
import scipy.sparse.linalg
from scipy.special import erfc

try:                                   # effective even if numpy is loaded
    from threadpoolctl import threadpool_limits
    threadpool_limits(MAX_THREADS)
except ImportError:
    pass

#
# HELPERS (provided)
#

def check(label, value, reference, tol):
    """
    Compare a computed value with a reference value and print PASS or FAIL.

    Returns True when |value - reference| <= tol.
    """
    ok = bool(np.all(np.abs(np.asarray(value) - np.asarray(reference)) <= tol))
    print('[%s] %-48s %+.10f  (ref. %+.10f, tol. %.1e)'
          % ('PASS' if ok else 'FAIL', label, np.real(value),
             np.real(reference), tol))
    return ok


def cubic_cell(a0):
    """Unit cell matrix of a cubic cell with edge a0."""
    return a0 * np.identity(3)


def cell_volume(M):
    """Volume Omega = |det M| of the unit cell."""
    return abs(np.linalg.det(M))


def grid_shape(N):
    """Turn an int or a triple into a tuple (N1, N2, N3)."""
    return (N, N, N) if np.isscalar(N) else tuple(int(n) for n in N)


def integrate(f, M):
    """Riemann sum Delta V * sum_i f(r_i) over the unit cell."""
    return cell_volume(M) / f.size * np.sum(f)


def minimum_image(r, R, M):
    """
    Displacement vectors r - R reduced to the nearest periodic image.
    r has shape (..., 3), R shape (3,).
    """
    s = (r - R) @ np.linalg.inv(M)      # fractional coordinates
    s -= np.rint(s)
    return s @ M


def gaussian_density(r, M, centre, alpha):
    """
    Normalized Gaussian charge cloud (alpha/pi)^(3/2) exp(-alpha |r - R|^2),
    centred at R and made periodic via the minimum-image convention.
    For the wave function g(r) of the primer, rho = g^2 has alpha = 2.
    """
    d = minimum_image(r, centre, M)
    return (alpha / np.pi)**1.5 * np.exp(-alpha * np.einsum('...i,...i', d, d))


def integer_vectors(nmax, include_zero=False):
    """
    All integer triples (n1, n2, n3) with |n_i| <= nmax[i], as an (n, 3)
    array, optionally excluding (0, 0, 0).
    """
    axes = [np.arange(-int(n), int(n) + 1) for n in nmax]
    m = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    return m if include_zero else m[np.any(m != 0, axis=1)]


#
# SECTION 2 -- PLANE WAVES AND UNIT CELLS
#

def realspace_grid(M, N):
    """
    [Exercise 2.1] Real-space sampling points.

    Returns r with shape (N1, N2, N3, 3) such that
    r[i, j, k] = (i/N1) a1 + (j/N2) a2 + (k/N3) a3.
    """
    N = grid_shape(N)
    s = [np.arange(n) / n for n in N]
    S = np.stack(np.meshgrid(*s, indexing='ij'), axis=-1)
    return S @ M


def gvectors(M, N):
    """
    [Exercise 2.1] Plane-wave vectors in numpy.fft order.

    Returns (G, G2): G with shape (N1, N2, N3, 3) and G2 = |G|^2 with shape
    (N1, N2, N3).
    """
    N = grid_shape(N)
    B = 2.0 * np.pi * np.linalg.inv(M.T)            # rows are b1, b2, b3
    m = [np.fft.fftfreq(n, d=1.0 / n) for n in N]   # integers 0,1,..,-1
    mm = np.stack(np.meshgrid(*m, indexing='ij'), axis=-1)
    G = mm @ B
    return G, np.einsum('...i,...i', G, G)


def fft_pw(f, M):
    """[Exercise 2.1] Expansion coefficients u~(G) = C_t * FFT[u(r)]."""
    return np.sqrt(cell_volume(M)) / f.size * np.fft.fftn(f)


def ifft_pw(c, M):
    """[Exercise 2.1] Real-space values u(r) = IFFT[u~(G)] / C_t."""
    return c.size / np.sqrt(cell_volume(M)) * np.fft.ifftn(c)


def kinetic_energy(c, G2):
    """[Exercise 2.2] Kinetic energy 1/2 sum_G |c(G)|^2 |G|^2 of one orbital."""
    return 0.5 * np.sum(np.abs(c)**2 * G2)


#
# SECTION 3 -- POTENTIALS
#

def hartree_potential(rho, G2):
    """
    [Exercise 3.1] Solve Poisson's equation for the (electron) density rho.

    The G = 0 coefficient is set to zero. Because a forward and an inverse
    transform are combined, numpy.fft can be used directly.
    """
    rhog = np.fft.fftn(rho)
    vg = np.zeros_like(rhog)
    nz = G2 > 0
    vg[nz] = 4.0 * np.pi * rhog[nz] / G2[nz]
    return np.fft.ifftn(vg).real


def external_potential(G, G2, M, positions, charges):
    """
    [Exercise 3.2] Potential energy of an electron in the field of point
    nuclei with charges Z_j at positions R_j (G = 0 term discarded).
    """
    Omega = cell_volume(M)
    sf = np.exp(-1j * G @ np.asarray(positions).T) @ np.asarray(charges)
    vg = np.zeros_like(sf)
    nz = G2 > 0
    vg[nz] = -4.0 * np.pi / np.sqrt(Omega) * sf[nz] / G2[nz]
    return ifft_pw(vg, M).real


RHO_FLOOR = 1e-20   # densities below this value are treated as vacuum


def lda_x(rho):
    """
    [Exercise 3.4] Slater exchange: energy per electron eps_x and potential
    v_x = 4/3 eps_x.
    """
    rho = np.maximum(rho, RHO_FLOOR)
    ex = -0.75 * (3.0 / np.pi)**(1.0 / 3.0) * rho**(1.0 / 3.0)
    return ex, 4.0 / 3.0 * ex


VWN_PARAMETERS = {
    # A, x0, b, c  (A in Rydberg units, hence the factor A/2 below)
    'vwn5': (0.0621814, -0.10498, 3.72744, 12.9352),
    'vwn3': (0.0621814, -0.409286, 13.0720, 42.7198),
}
VWN_DEFAULT = 'vwn5'


def vwn_c(rho, variant=VWN_DEFAULT):
    """
    [provided] VWN correlation (appendix of the primer): energy per electron
    eps_c and potential v_c = eps_c - r_s/3 d eps_c / d r_s.
    """
    A, x0, b, c = VWN_PARAMETERS[variant]
    rho = np.maximum(rho, RHO_FLOOR)
    rs = (3.0 / (4.0 * np.pi * rho))**(1.0 / 3.0)
    x = np.sqrt(rs)
    X = x**2 + b * x + c
    X0 = x0**2 + b * x0 + c
    Q = np.sqrt(4.0 * c - b**2)
    atan = np.arctan(Q / (2.0 * x + b))
    ec = A / 2.0 * (np.log(x**2 / X) + 2.0 * b / Q * atan
                    - b * x0 / X0 * (np.log((x - x0)**2 / X)
                                     + 2.0 * (b + 2.0 * x0) / Q * atan))
    # d eps_c / dx, then v_c = eps_c - x/6 d eps_c/dx (since r_s = x^2)
    dX = 2.0 * x + b
    decdx = A / 2.0 * (2.0 / x - dX / X - 4.0 * b / (dX**2 + Q**2)
                       - b * x0 / X0 * (2.0 / (x - x0) - dX / X
                                        - 4.0 * (b + 2.0 * x0) / (dX**2 + Q**2)))
    return ec, ec - x / 6.0 * decdx


def lda_xc(rho, variant=VWN_DEFAULT):
    """[Exercise 3.4] Slater exchange + VWN correlation: (eps_xc, v_xc)."""
    ex, vx = lda_x(rho)
    ec, vc = vwn_c(rho, variant)
    return ex + ec, vx + vc


#
# SECTION 4 -- EWALD SUMMATION
#

def ewald_parameters(M, gcut=2.0, gamma=1e-8):
    """
    [provided] Splitting parameter alpha and the ranges n_max (real space)
    and m_max (reciprocal space) following the recipe of the primer
    (orthogonal cells).
    """
    alpha = -0.25 * gcut**2 / np.log(gamma)
    amag = np.linalg.norm(M, axis=1)
    bmag = np.linalg.norm(2.0 * np.pi * np.linalg.inv(M.T), axis=1)
    nmax = np.rint(np.sqrt(-0.5 * np.log(gamma)) / np.sqrt(alpha) / amag + 1.5)
    mmax = np.rint(gcut / bmag + 1.5)
    return alpha, nmax.astype(int), mmax.astype(int)


def ewald_energy(M, positions, charges, gcut=2.0, gamma=1e-8,
                 return_terms=False):
    """
    [Exercise 4.1] Electrostatic energy of a periodic array of point charges
    in a uniform neutralizing background, E = E_sr + E_lr - E_s - E_en.
    """
    R = np.asarray(positions, dtype=float)
    q = np.asarray(charges, dtype=float)
    Omega = cell_volume(M)
    alpha, nmax, mmax = ewald_parameters(M, gcut, gamma)

    # short-range part, real space
    T = integer_vectors(nmax, include_zero=True) @ M
    is_zero = np.all(T == 0, axis=1)
    Esr = 0.0
    for i in range(len(q)):
        for j in range(len(q)):
            d = np.linalg.norm(R[i] - R[j] + T, axis=1)
            mask = ~is_zero if i == j else np.ones(len(T), dtype=bool)
            Esr += 0.5 * q[i] * q[j] * np.sum(
                erfc(np.sqrt(alpha) * d[mask]) / d[mask])

    # long-range part, reciprocal space (long-range term with cosines, Section 4)
    B = 2.0 * np.pi * np.linalg.inv(M.T)
    G = integer_vectors(mmax) @ B
    G2 = np.einsum('ij,ij->i', G, G)
    pre = 2.0 * np.pi / Omega * np.exp(-0.25 * G2 / alpha) / G2
    Elr = 0.0
    for i in range(len(q)):
        for j in range(len(q)):
            Elr += q[i] * q[j] * np.sum(pre * np.cos(G @ (R[i] - R[j])))

    Es = np.sqrt(alpha / np.pi) * np.sum(q**2)
    Een = np.pi / (2.0 * alpha * Omega) * np.sum(q)**2

    E = Esr + Elr - Es - Een
    if return_terms:
        return E, {'sr': Esr, 'lr': Elr, 's': Es, 'en': Een}
    return E


#
# SECTION 5 -- SELF-CONSISTENT FIELD
#

def density_from_orbitals(C, M, occupation=2.0):
    """
    [Exercise 5.1] Electron density rho(r) = occ * sum_i |psi_i(r)|^2 from
    the plane-wave coefficients C[i] of the occupied orbitals.
    """
    return occupation * sum(np.abs(ifft_pw(c, M))**2 for c in C)


def energy_components(C, M, G2, vext, e_ewald, variant=VWN_DEFAULT):
    """
    [Exercise 5.1] Total energy and its components (doubly occupied orbitals).
    """
    rho = density_from_orbitals(C, M)
    vh = hartree_potential(rho, G2)
    exc, _ = lda_xc(rho, variant)
    terms = {
        'kinetic': 2.0 * sum(kinetic_energy(c, G2) for c in C),
        'nuclear': integrate(vext * rho, M),
        'hartree': 0.5 * integrate(vh * rho, M),
        'xc': integrate(exc * rho, M),
        'ewald': e_ewald,
    }
    terms['total'] = sum(terms.values())
    return terms


def effective_potential(rho, vext, G2, variant=VWN_DEFAULT):
    """[Exercise 5.2] v_eff = v_ext + v_H[rho] + v_xc[rho]."""
    return vext + hartree_potential(rho, G2) + lda_xc(rho, variant)[1]


def apply_hamiltonian(c, veff, G2, M):
    """
    [Exercise 5.2] H c = 1/2 |G|^2 c + FFT_pw[ IFFT_pw[c] * v_eff ].
    """
    return 0.5 * G2 * c + fft_pw(ifft_pw(c, M) * veff, M)


def hamiltonian_operator(veff, G2, M):
    """
    [Exercise 5.2] The Hamiltonian as a matrix-free scipy LinearOperator that
    acts on flattened coefficient vectors.
    """
    shape, n = G2.shape, G2.size

    def matvec(x):
        return apply_hamiltonian(x.reshape(shape), veff, G2, M).ravel()

    return scipy.sparse.linalg.LinearOperator((n, n), matvec=matvec,
                                              dtype=np.complex128)


def lowest_eigenstates_arpack(veff, G2, M, nstates, tol=1e-10):
    """
    [Exercise 5.2] Lowest `nstates` eigenpairs with ARPACK (eigsh), i.e. the
    implicitly restarted Arnoldi method of the primer.

    Returns (eigenvalues, C) with C of shape (nstates, N1, N2, N3).
    """
    H = hamiltonian_operator(veff, G2, M)
    e, v = scipy.sparse.linalg.eigsh(H, k=nstates, which='SA', tol=tol)
    order = np.argsort(e)
    return e[order], v[:, order].T.reshape((nstates,) + G2.shape)


def lowest_eigenstates(veff, G2, M, nstates, C0=None, tol=1e-8, maxiter=400):
    """
    [provided] Lowest `nstates` eigenpairs with a preconditioned BLOCK method
    (LOBPCG, closely related to the block Davidson method of the primer).

    A block of trial vectors, one per state, is iterated at once, so that
    degenerate states are found reliably (Exercise 5.2 shows why a single
    Krylov sequence struggles with them). The preconditioner divides each
    residual by 1 + |G|^2/2, damping the high-kinetic-energy plane waves. The
    previous orbitals C0, if given, serve as starting block.

    Returns (eigenvalues, C) with C of shape (nstates, N1, N2, N3).
    """
    shape, n = G2.shape, G2.size
    H = hamiltonian_operator(veff, G2, M)
    d = 1.0 + 0.5 * G2.ravel()
    P = scipy.sparse.linalg.LinearOperator(
        (n, n), matvec=lambda x: x.ravel() / d, matmat=lambda X: X / d[:, None],
        dtype=np.complex128)
    if C0 is None:      # smooth random start: low-|G| plane waves only
        rng = np.random.default_rng(0)
        X = (rng.normal(size=(n, nstates)) + 1j * rng.normal(size=(n, nstates)))
        X *= np.exp(-0.5 * G2.ravel())[:, None]
    else:
        X = np.asarray(C0).reshape(nstates, n).T.copy()
    with warnings.catch_warnings():     # 'tolerance not reached' near 1e-8
        warnings.simplefilter('ignore', UserWarning)
        e, v = scipy.sparse.linalg.lobpcg(H, X, M=P, largest=False, tol=tol,
                                          maxiter=maxiter)
    order = np.argsort(e)
    return e[order], v[:, order].T.reshape((nstates,) + shape)


#
# SECTION 6 -- WORKING WITH PYPWDFT (provided)
#

ANGSTROM = 1.8897259886           # bohr per angstrom
# PBE equilibrium bond lengths in angstrom (primer, Section 6)
DIATOMICS = {'H2': ('H', 'H', 0.7500), 'CO': ('C', 'O', 1.1352)}


def diatomic(name, L, bond=None):
    """
    [provided] PyPWDFT Structure of a diatomic molecule along z, centred in a
    cubic cell of edge L (bohr). `bond` overrides the bond length (angstrom).
    """
    from pypwdft import Structure
    a, b, d = DIATOMICS[name]
    d = d if bond is None else bond
    return Structure([a, b], [[0.0, 0.0, 0.0], [0.0, 0.0, d * ANGSTROM]], L)


def from_pypwdft(field):
    """
    [provided] Convert a PyPWDFT grid array, indexed [z, y, x], to the
    convention of pwlib, [x, y, z]. Works for a single field and for a stack
    of orbitals (leading axis).
    """
    return np.swapaxes(np.asarray(field), -1, -3)


#
# SECTION 7 -- PSEUDOPOTENTIALS
#

# GTH-PBE pseudopotential of oxygen (Krack), as printed in the primer
GTH_O = {'Zion': 6.0, 'rloc': 0.24455430, 'C': (-16.66721480, 2.48731132, 0.0, 0.0),
         'rs': 0.22095592, 'hs': 18.33745811}


def gth_local_r(r, Zion, rloc, C):
    """[provided] GTH local potential nu_loc(r) in real space (primer, Section 7)."""
    from scipy.special import erf
    r = np.asarray(r, dtype=float)
    x = r / rloc
    poly = sum(c * x**(2 * k) for k, c in enumerate(C))
    with np.errstate(divide='ignore', invalid='ignore'):
        coul = np.where(r > 0, -Zion / r * erf(r / (np.sqrt(2.0) * rloc)),
                        -Zion * np.sqrt(2.0 / np.pi) / rloc)
    return coul + np.exp(-0.5 * x**2) * poly


def gth_local_g(G2, Zion, rloc, C):
    """
    [Exercise 7.1] Fourier transform of the GTH local potential of a single
    atom at the origin (primer, Appendix B). At G = 0, the divergent Coulomb part
    is discarded and the finite constant alpha is returned instead.
    """
    G2 = np.asarray(G2, dtype=float)
    y = G2 * rloc**2
    C1, C2, C3, C4 = C
    short = (2.0 * np.pi)**1.5 * rloc**3 * np.exp(-0.5 * y) * (
        C1 + C2 * (3 - y) + C3 * (15 - 10 * y + y**2)
        + C4 * (105 - 105 * y + 21 * y**2 - y**3))
    out = np.empty_like(G2)
    nz = G2 > 0
    out[nz] = -4.0 * np.pi * Zion / G2[nz] * np.exp(-0.5 * y[nz]) + short[nz]
    alpha = (2.0 * np.pi * Zion * rloc**2
             + (2.0 * np.pi)**1.5 * rloc**3 * (C1 + 3 * C2 + 15 * C3 + 105 * C4))
    out[~nz] = alpha
    return out


def gth_external_potential(G, G2, M, positions, params):
    """
    [Exercise 7.1] Local pseudopotential of a set of atoms on the grid
    (primer, Section 7); `params` is a list with one parameter dict per atom.
    """
    Omega = cell_volume(M)
    vg = np.zeros(G2.shape, dtype=complex)
    for R, p in zip(np.asarray(positions, dtype=float), params):
        vg += gth_local_g(G2, p['Zion'], p['rloc'], p['C']) * np.exp(-1j * G @ R)
    return ifft_pw(vg / np.sqrt(Omega), M).real


def gth_projector_s_g(G, G2, M, R, rs):
    """
    [Exercise 7.2] Plane-wave coefficients of the s projector p^0_1 of an atom
    at R (appendix: projectors, with Y_00 = 1/sqrt(4 pi)).
    """
    f = 4.0 * np.pi**1.25 * np.sqrt(2.0 * rs**3) * np.exp(-0.5 * G2 * rs**2)
    return f / np.sqrt(4.0 * np.pi) * np.exp(-1j * G @ np.asarray(R)) \
        / np.sqrt(cell_volume(M))
