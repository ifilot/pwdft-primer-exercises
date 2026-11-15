# -*- coding: utf-8 -*-
"""
Derive the student version exercises/pwlib.py from the reference version
solutions/pwlib.py: the bodies of all functions that students write are
replaced by hints and a NotImplementedError. Run after every change to the
reference module.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'solutions' / 'pwlib.py'
DST = ROOT / 'exercises' / 'pwlib.py'

HINTS = {
    'realspace_grid': """
    # Hints:
    #  * np.arange(n) / n gives the fractional coordinates along one axis.
    #  * np.meshgrid(..., indexing='ij') keeps the first index along a1.
    #  * Stack the three fractional grids along a last axis (np.stack) and
    #    multiply by M: fractional coordinates @ M = Cartesian coordinates.""",
    'gvectors': """
    # Hints:
    #  * Reciprocal lattice vectors as rows: B = 2 pi (M^T)^-1.
    #  * np.fft.fftfreq(n, d=1/n) yields the integers 0, 1, ..., -1 in the
    #    order used by numpy.fft.
    #  * Build integer triples with np.meshgrid(..., indexing='ij'), multiply
    #    by B, and compute |G|^2 with np.einsum('...i,...i', G, G).""",
    'fft_pw': """
    # Hint: C_t = sqrt(Omega) / (N1 N2 N3); f.size = N1 N2 N3.""",
    'ifft_pw': """
    # Hint: divide by the same constant C_t that fft_pw multiplies with.""",
    'kinetic_energy': """
    # Hint: one line, cf. the expression for E_kin in Section 2.""",
    'hartree_potential': """
    # Hints:
    #  * Transform rho with np.fft.fftn, multiply by 4 pi / |G|^2, transform
    #    back with np.fft.ifftn and return the real part.
    #  * Avoid the division by zero: compute only where G2 > 0 and leave the
    #    G = 0 coefficient at zero.""",
    'external_potential': """
    # Hints:
    #  * Structure factor: sum_j Z_j exp(-i G.R_j). With G of shape
    #    (N1, N2, N3, 3) and positions of shape (natom, 3), G @ positions.T
    #    has shape (N1, N2, N3, natom).
    #  * Multiply by -4 pi / (sqrt(Omega) |G|^2) for G != 0 and transform with
    #    your ifft_pw. Return the real part.""",
    'lda_x': """
    # Hints:
    #  * eps_x = -3/4 (3/pi)^(1/3) rho^(1/3) and v_x = 4/3 eps_x.
    #  * First replace rho by np.maximum(rho, RHO_FLOOR): why is that needed?""",
    'lda_xc': """
    # Hint: combine lda_x with the provided vwn_c.""",
    'density_from_orbitals': """
    # Hint: transform every orbital to real space with ifft_pw.""",
    'energy_components': """
    # Hints:
    #  * Mind the factor 2 (double occupation) and the factor 1/2 in front of
    #    the Hartree energy.
    #  * Return a dict with keys 'kinetic', 'nuclear', 'hartree', 'xc',
    #    'ewald' and 'total'.""",
    'effective_potential': """
    # Hint: one line, using hartree_potential and lda_xc.""",
    'apply_hamiltonian': """
    # Hint: the kinetic part acts in reciprocal space, the potential in real
    # space; use fft_pw and ifft_pw.""",
    'hamiltonian_operator': """
    # Hints:
    #  * scipy works with flat vectors of length n = G2.size: reshape the
    #    input to G2.shape, apply the Hamiltonian, and flatten (ravel) again.
    #  * scipy.sparse.linalg.LinearOperator((n, n), matvec=..., dtype=complex)""",
    'gth_local_g': """
    # Hints:
    #  * Use y = G2 * rloc^2 and the expression in the appendix of the primer.
    #  * Evaluate the Coulomb part only where G2 > 0; at G = 0 return the
    #    finite constant alpha (also in the appendix).""",
    'gth_external_potential': """
    # Hint: as external_potential, but with gth_local_g(G2, ...) of every atom
    # in place of -4 pi Z / |G|^2, and keep the G = 0 coefficient.""",
    'gth_projector_s_g': """
    # Hints:
    #  * f(G) = 4 pi^(5/4) sqrt(2 rs^3) exp(-G^2 rs^2 / 2) for l = 0, i = 1.
    #  * Multiply by Y_00 = 1/sqrt(4 pi), the phase factor exp(-i G.R) and
    #    1/sqrt(Omega).""",
    'lowest_eigenstates_arpack': """
    # Hints:
    #  * scipy.sparse.linalg.eigsh(H, k=nstates, which='SA', tol=tol)
    #  * eigsh returns the eigenvectors as COLUMNS of a (n, nstates) array;
    #    sort by eigenvalue and reshape to (nstates, N1, N2, N3).""",
}

EWALD_STUB = '''
    R = np.asarray(positions, dtype=float)
    q = np.asarray(charges, dtype=float)
    Omega = cell_volume(M)
    alpha, nmax, mmax = ewald_parameters(M, gcut, gamma)

    # lattice vectors n (including n = 0) for the real-space sum
    T = integer_vectors(nmax, include_zero=True) @ M
    # plane-wave vectors G (excluding G = 0) for the reciprocal-space sum
    B = 2.0 * np.pi * np.linalg.inv(M.T)
    G = integer_vectors(mmax) @ B

    # TODO: short-range energy E_sr (eq. ewald_sr). Loop over all pairs
    # (i, j); for each pair compute the distances |R_i - R_j + n| for all n
    # at once and skip the term n = 0 when i == j. Use scipy's erfc.
    Esr = 0.0

    # TODO: long-range energy E_lr (eq. long-range-cos)
    Elr = 0.0

    # TODO: self-interaction E_s (eq. ewald_self) and electroneutrality
    # term E_en (eq. ewald_en)
    Es = 0.0
    Een = 0.0

    raise NotImplementedError('Exercise 4.1: implement the four terms, then '
                              'remove this line')
    E = Esr + Elr - Es - Een
    if return_terms:
        return E, {'sr': Esr, 'lr': Elr, 's': Es, 'en': Een}
    return E
'''

HEADER_NOTE = '''
# ----------------------------------------------------------------------------
# STUDENT VERSION. Functions marked [Exercise x.y] are yours to write; the
# others are provided. Fallen behind? Copy the function from
# solutions/pwlib.py so that you can continue with the later exercises.
# ----------------------------------------------------------------------------
'''


def main():
    src = SRC.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    replacements = []          # (first body line after docstring, last line, text)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        name = node.name
        if name not in HINTS and name != 'ewald_energy':
            continue
        doc = node.body[0]
        start = doc.end_lineno            # 1-based last line of the docstring
        end = node.end_lineno
        exercise = ast.get_docstring(node).split(']')[0].strip('[')
        if name == 'ewald_energy':
            body = EWALD_STUB.rstrip('\n')
        else:
            body = (HINTS[name].rstrip('\n')
                    + "\n    raise NotImplementedError('%s')" % exercise)
        replacements.append((start, end, body))

    for start, end, body in sorted(replacements, reverse=True):
        lines[start:end] = body.split('\n')[1:] if body.startswith('\n') else body.split('\n')

    out = '\n'.join(lines) + '\n'
    out = out.replace('This is the REFERENCE version.', 'This is the STUDENT version.')
    # insert the note after the module docstring
    end_doc = out.index('"""', out.index('"""') + 3) + 3
    out = out[:end_doc] + '\n' + HEADER_NOTE + out[end_doc:]
    DST.write_text(out)
    print('Wrote', DST)


if __name__ == '__main__':
    main()
