# Exercises for *Plane-wave density functional theory: a primer* — draft

**Status (2026-11-15): integrated into the reader** as `exci`/`exct` boxes in
an "Exercises" subsection at the end of Sections 2–7. From now on the reader is
the authoritative text; this file keeps the author notes, the reference values
from the solution scripts and the history of changes. Reader equations, figures
and sections are referred to by their LaTeX labels in square brackets, e.g.
[eq:gvectors] is `\cref{eq:gvectors}`.

---

## Notes for the author (not for the reader)

**Two kinds of exercise.** Each box is labelled either *Exercise* or *Challenge*;
`\begin{exct}[Challenge]` already supports this, and numbering is shared.

* **Exercise** (light, 20–40 min): a skeleton script in `exercises/` contains
  the checks; students write a few functions in `exercises/pwlib.py` and fill
  in short TODO blocks. Every check prints `PASS` or `FAIL`.
* **Challenge** (heavy, 1.5–4 h): an outline only; students design the
  calculation and interpret the results. Challenges build on the Exercises.

**Suggested winter-school path (about 3.5 h):** 2.1, 2.2, 3.1, 3.2, 4.1, 4.2,
5.1, 5.2, and 5.5 if time permits. The Challenges complete the set for course
use.

**Design principles.**

1. Students build their own small plane-wave code, `pwlib.py`, function by
   function. Later exercises import it; anyone who falls behind copies the
   missing function from `solutions/pwlib.py`.
2. Wherever possible, the first check reproduces a number printed in the
   reader (Fig. 4 data, the periodic-system formulas of Section 3, Table
   [tab:background], the Madelung constants, the Ewald parameter recipe); the
   exercise then goes one step further. Exceptions: Section 5 has no printed
   reference numbers for a complete calculation, so self-consistency checks
   and the CH₄ reference data take their place.
3. The atoms and charge clouds are placed off-centre where that matters
   (Exercise 3.2), so that sign errors cannot hide behind symmetry.
4. All reference numbers below come from the solution scripts in
   `solutions/`, which run in a few minutes in total.

**Points to decide or check.**

* *VWN variant (decided: VWN5).* The exercises use VWN5
  (`pwlib.VWN_DEFAULT`), as PyPWDFT does. The appendix of the reader still
  states that the primer uses the RPA fit (VWN3); that sentence should be
  updated once the reader's figures are computed with VWN5.
* *Degenerate states and ARPACK.* Exercise 5.2 shows that `eigsh` (one
  Krylov sequence) can miss a member of a degenerate set: it consistently
  misses one of the three 2p states of a hydrogen atom on a cubic grid. The
  SCF in Challenge 5.4 therefore uses a preconditioned block solver (LOBPCG),
  provided in pwlib. The reader currently presents `eigsh` without this
  caveat; one or two sentences in the Arnoldi box or the Davidson paragraph
  might be worthwhile.
* *Charge sloshing.* [sec:mixing] says that sloshing becomes worse in larger
  cells. For a small molecule in vacuum we do not see this: H₂ converges in the
  same number of iterations in cells of 10, 15 and 20 bohr. The statement holds
  for systems that fill the cell (metals, extended systems), whose density
  responds strongly at long wavelengths. A qualifying half-sentence in the
  reader may help; Challenge 5.4(c) now asks students to test the claim.
* *Sections 6 and 7* use PyPWDFT 0.3.0 (PyPI). Its grids are indexed
  `[z, y, x]`, whereas pwlib uses `[x, y, z]`; the helper `pwlib.from_pypwdft`
  converts, and the exercises point this out where it matters. If you prefer
  one convention throughout, pwlib could be switched to `[z, y, x]`.
* *Section 8* still points to the old repository `hsl-pwdft-exercises`.

**Changes with respect to the old exercise set.**

| Old | New | Main changes |
|-----|-----|--------------|
| 1, 2 | 2.1, 2.2 | Students now build the grids themselves (the old template gave them away); plane-wave test with a single coefficient; H 1s reproduces Fig. 4; exact kinetic energy of the CO 1π orbitals instead of "approximately 2.0 Ht"; fixed the ⟨6\|6⟩ bug and the xy/xz plane mismatch. |
| 3, 4 | 3.1, 3.2, 3.3 | Compared with the periodic-system formulas of Section 3 instead of "will be fixed in the next exercise"; off-centre nucleus (the old statement had exp(+iG·R), a sign error masked by the central position). |
| 5 | 4.1–4.3 | Students implement the four terms (the old template gave the full routine); validated by Madelung constants and α-independence; G_cut is no longer called "E_cut in Ht". |
| 6 | 5.1 | Fixed the C coordinate (z = 3.7097 in the old PDF); new reference data from our own SCF. |
| 7 | 5.2, 5.3 | Replaced: the old Fock matrix was in a non-orthogonal AO basis and was diagonalized without its overlap matrix (eigenvalues −23.4/−13.1 Ht instead of −20.4/−11.1 Ht). Now: a matrix-free Hamiltonian, as in Section 5. |
| 8 | 5.5, 5.6 | Closed-form phase instead of differential evolution; degenerate states treated properly; no PyPWDFT needed. |
| — | 2.3, 5.4 | New: cut-off scaling with Z (bridge to Section 7); a student-built SCF. |

---

## Practical information (for the reader)

All exercises use Python 3 with NumPy, SciPy and Matplotlib. Download the
repository, open the folder `exercises/`, and run each script from there, e.g.
`python ex2_1_grids.py`. During the exercises you build a small module,
`pwlib.py`, step by step; every function you have to write is marked with the
exercise in which you write it. Each script checks your results and prints
`PASS` or `FAIL`. A reference solution is available in `solutions/`; consult it
after you have made a serious attempt.

Two conventions recur throughout:

* A scalar field on the grid is an array `f[i, j, k]` holding the value at
  r = (i/N₁) a₁ + (j/N₂) a₂ + (k/N₃) a₃. For a cubic cell, x is thus the first
  index and z the last.
* Plane-wave coefficients are stored in the order of `numpy.fft`: G = 0 first,
  then the positive, then the negative frequencies (Section 2,
  [sec:planewaves]).

---

## Section 2 — Plane waves and unit cells

### Exercise 2.1 — Grids, plane waves and the FFT convention

*Exercise · about 30 min · `ex2_1_grids.py` · write `realspace_grid`,
`gvectors`, `fft_pw`, `ifft_pw`*

Every plane-wave calculation rests on two grids, the real-space sampling
points and the plane-wave vectors, and on a Fourier transform that maps one
onto the other. Get either the ordering or the normalization wrong and every
later result is wrong, often without any visible symptom. This exercise
therefore tests the foundations in isolation, using a function whose
transform you know exactly: a single plane wave.

a) Implement `realspace_grid(M, N)` and `gvectors(M, N)` for a general unit
   cell M [eq:gvectors] and the reciprocal lattice vectors
   [eq:reciproc-latv1]. Order the G-vectors as `numpy.fft` does (see
   `numpy.fft.fftfreq`). For a0 = 1 and N = 8, compare the values of G_x with
   [fig:sampling]. Explain why the range of G_x spans 2π(N − 1)/a0 rather than
   the 2πN/a0 mentioned in the caption.
b) Implement `fft_pw` and `ifft_pw` using the constant C_t [eq:ct]. Sample the
   plane wave φ(G₀, r) [eq:pwexpr] with G₀ = (1, −2, 3)·2π/a0 in a cubic cell
   with a0 = 10 and N = 32. **Verify** that exactly one coefficient is
   non-zero, that it equals 1, and that it sits at G₀. Which property of the
   plane-wave basis does this demonstrate?
c) **Verify** that `ifft_pw(fft_pw(f))` returns f for a random field f.
d) *(Optional)* Repeat (b) with m = (N/2 + 3, 0, 0). Where does the
   coefficient end up, and why? You will meet this effect, aliasing, again in
   [sec:dualgrid].

### Exercise 2.2 — One orbital, two representations

*Exercise · about 40 min · `ex2_2_orbitals.py` · write `kinetic_energy`*

An orbital can be stored as values on a grid or as plane-wave coefficients.
Some quantities are easier to compute in one representation than in the
other, but both must give the same answer. In this exercise you first
reproduce the hydrogen demonstration of Section 2 and then turn to a
molecule.

a) Sample the 1s orbital of hydrogen [eq:horb_1s] (a0 = 10, N = 32,
   R = (5, 5, 5)). Calculate (i) its norm by numerical integration in real
   space, (ii) the sum of its squared plane-wave coefficients, and (iii) its
   kinetic energy [eq:kin_en]. **Check:** (i) and (ii) agree to machine
   precision, and reproduce the values behind [fig:ekin_convergence]: 1.00077013
   and 0.51662733 Ht. Why does the norm exceed one?
b) The file `data/co_1pi.npz` contains the doubly degenerate 1π orbitals of CO
   (HF/STO-3G), sampled with N = 64 in a cell with a0 = 10 bohr. The molecule
   lies along z, and the file also lists the atomic positions and the exact
   kinetic energy in the Gaussian basis. Make contour plots of both orbitals in
   the xz-plane through the molecular axis and in the xy-plane halfway between
   the nuclei. Calculate the 2 × 2 overlap matrix in real space and from the
   plane-wave coefficients, and the kinetic energy of both orbitals.
   **Check:** the two overlap matrices agree; the kinetic energy, 2.000888 Ht,
   deviates from the exact value of 2.000738 Ht by 0.15 mHt. **Interpret:** why
   do the two orbitals have exactly the same kinetic energy, and why does this
   Gaussian-based orbital converge much faster than the hydrogen 1s orbital in
   (a)?
c) Shift ψ₅ over a whole number of grid points with `numpy.roll`.
   **Verify** that each coefficient is multiplied by exp(−iG·d), and that
   |c(G)| and the kinetic energy are unchanged (cf. Section 2.5).

### Challenge 2.3 — Cusps, cell size and the cut-off energy

*Challenge · about 2 h · `chal2_3_cutoff.py`*

[fig:ekin_convergence] shows two sources of error: a finite grid and a finite
cell. In this challenge you separate them, and you find out how the required
cut-off energy grows with the nuclear charge. That scaling is the reason for
the pseudopotentials of Section 7.

a) Reproduce [fig:ekin_convergence] for N = 8 to 128 (and 256 if your memory
   allows). **Check:** at N = 128 you should find a norm of 0.99919444 and a
   kinetic energy of 0.49990246 Ht.
b) Calculate the fraction of the 1s orbital that lies outside the cell, e.g. by
   Gauss–Legendre quadrature over one octant. Show that it explains the plateau
   in (a) (expected: 8.1 × 10⁻⁴).
c) Repeat (a) for the Gaussian g(r) of Section 3. How many points per
   direction are needed for machine precision, and why so few?
d) Now apply a spherical cut-off: keep only the plane waves with
   ½|G|² ≤ E_cut (Section 2.4, [sec:truncation]). To separate this from grid
   and cell-size errors, use the analytical coefficients of the hydrogen-like
   orbital √(Z³/π) exp(−Zr),
   c(G) = 8√(πZ⁵) / [(Z² + |G|²)² √Ω],
   and measure the error relative to the complete sum. Determine (i) how the
   number of plane waves grows with E_cut, and (ii) the cut-off needed for a 1%
   error in the kinetic energy, for Z = 1, 2 and 3. **Interpret:** you should
   find N_pw ∝ E_cut^{3/2} and E_cut ≈ 23 Z² Ht. What would this imply for the
   1s electrons of oxygen (Z = 8), and how many plane waves would a 10 bohr cell
   then need?

---

## Section 3 — Potentials

### Exercise 3.1 — The Hartree potential of a Gaussian charge cloud

*Exercise · about 30 min · `ex3_1_hartree.py` · write `hartree_potential`*

Solving Poisson's equation in reciprocal space takes three lines of code. The
subtle part is the discarded G = 0 term, which turns a single charge cloud into
a periodic array of clouds in a neutralizing background. Your implementation is
correct when it reproduces the energy of *that* system, [eq:erep_periodic],
rather than the isolated value of 2/√π.

a) Implement `hartree_potential` following the recipe below
   [eq:hartree-pot]. For the Gaussian density ρ = |g|² at the centre of a cell
   with a0 = 10, calculate E_rep = ∫ν_H ρ dr for N = 16, 24, 32 and 48.
   **Check:** from N = 32 on, E_rep agrees with [eq:erep_periodic] to within
   10⁻⁷ Ht (E_rep = 0.8477910 Ht), and E_rep/2 reproduces E_H = 0.4239 Ht of
   [tab:background].
b) **Verify** that the average of ν_H over the cell vanishes. Plot ν_H along
   a line through the centre of the cloud together with the potential of the
   isolated cloud, erf(√2 r)/r. **Interpret:** the two differ by about
   −0.282 Ht at the centre. Relate this offset to the terms of
   [eq:erep_periodic], and explain why ν_H curves upward towards the cell
   boundary instead of decaying as 1/r.

### Exercise 3.2 — The external potential and cancelling Madelung terms

*Exercise · about 30 min · `ex3_2_external.py` · write `external_potential`*

a) Implement `external_potential` [eq:nucpotpw] for an arbitrary set of
   nuclei. Place a proton at the *off-centre, off-grid* position
   R = (3.7, 5.2, 6.4) in a cell with a0 = 10 (N = 48), centre the Gaussian
   density on it, and calculate E_nuc. **Check:** agreement with
   [eq:epot_periodic] to within 10⁻⁶ Ht, and with [tab:background]
   (−1.3136 Ht).
b) **Show** that E_nuc + E_rep differs from the isolated value
   2(1 − √2)/√π only by 2π⟨r²⟩/(3a0³), both for a0 = 10 and a0 = 20 (N = 96).
   Which terms have cancelled, and why?
c) The skeleton can evaluate the structure factor with the wrong sign,
   exp(+iG·R). Compare E_nuc for both signs, for the off-centre proton and for
   a proton at the cell centre. **Interpret:** why can a sign error go
   unnoticed when every test places the nucleus at (5, 5, 5)?

### Challenge 3.3 — Measuring the Madelung constant

*Challenge · about 1.5 h · `chal3_3_madelung.py`*

The Madelung constant ξ is usually quoted, not measured. Here you extract it
from your own plane-wave calculations.

a) For a Gaussian cloud (α = 2) centred on a proton, calculate E_rep and E_nuc
   for cell edges a0 between 10 and 40 bohr at a fixed grid spacing of about
   0.31 bohr. Why must the spacing, rather than N, be kept fixed?
b) Fit the deviations from the isolated values to p/a0 + q/a0³ by linear least
   squares. **Check:** both fits give ξ = 2.837297 to six digits, and spread
   terms q = +4π⟨r²⟩/3 (E_rep) and −2π⟨r²⟩/3 (E_nuc).
c) Repeat for α = 1. Which fitted coefficient changes and which does not?
   Relate your answer to the statement in [sec:background] that, to leading
   order, the shift depends only on the amount of charge.
d) Plot |ΔE_rep|, |ΔE_nuc| and |ΔE_rep + ΔE_nuc| versus a0 on log–log axes and
   compare with [fig:electrostatics]b.
e) *(Optional, memory-intensive)* Repeat for the hydrogen 1s density, which
   needs a much finer grid. Why?

### Exercise 3.4 — The local density approximation

*Exercise · about 30 min · `ex3_4_lda.py` · write `lda_x`, `lda_xc` (`vwn_c`
is provided)*

a) Implement Slater exchange [eq:chap04:slater_exchange] with
   ν_x = (4/3) ε_x. **Derive** ∫ρ^{4/3} dr for the Gaussian density and
   **verify** your E_x (expected −0.3827516 Ht). Why does the function first
   replace ρ by `max(ρ, RHO_FLOOR)`?
b) A potential must be the functional derivative of its energy. **Verify**
   this numerically for ν_x and ν_xc [eq:lda_potential]: compare
   [E(ρ + εδρ) − E(ρ − εδρ)]/(2ε) with ∫ν δρ dr for the perturbation
   δρ = ρ cos(2πx/a0). *(Try also a broad Gaussian perturbation that does not
   vanish where ρ does, and explain why the agreement deteriorates.)*
c) A GGA needs ∇ρ, which follows from iG ρ̃(G) (Section 3.5). Compare the
   result with the analytical gradient for N = 32 and N = 48, and explain the
   difference in accuracy.
d) Which fraction of the grid points has ρ < 10⁻¹⁰, and how much do they
   contribute to E_xc? What does this mean for a GGA in the vacuum region?

---

## Section 4 — Ewald summation

### Exercise 4.1 — The four terms of the Ewald sum

*Exercise · about 40 min · `ex4_1_ewald.py` · complete `ewald_energy`
(`ewald_parameters` is provided)*

a) Complete the four terms [eq:ewald_sr], [eq:long-range-cos],
   [eq:ewald_self] and [eq:ewald_en] in `ewald_energy`. **Check** that the
   parameter recipe reproduces the values of n_max and m_max quoted in
   [sec:est_ewald].
b) One unit charge in a cubic cell has the energy −ξ/(2a0). **Check:** your
   code returns ξ = 2.837297.
c) Rock salt: in the conventional cubic cell (four Na⁺, four Cl⁻) the energy
   per ion pair is −M/d, with d the nearest-neighbour distance. **Check:**
   M = 1.747565. Why does E_en vanish here?
d) For CO in a 10 bohr cell (positions in the skeleton), tabulate α, the four
   terms and E_Ewald for G_cut = 0.5 to 6 bohr⁻¹. **Check:** the individual
   terms vary by tens of Hartrees, whereas E_Ewald = −5.99562232 Ht is constant
   to 10⁻⁸ Ht. **Interpret:** why is the result negative although both
   nuclei are positive? (See [sec:background].)

### Exercise 4.2 — Why the background charge does no harm

*Exercise · about 20 min · `ex4_2_background.py`*

Reproduce [tab:background] for L = 10 (N = 48) and L = 20 (N = 96) with your
own pwlib. **Check:** all entries agree to the printed digits; E_Ewald equals
−ξ/(2L); the sum equals the isolated value −1.0316 Ht for both cells.
**Interpret:** explain in two or three sentences why you should never compare
an individual electrostatic term of a plane-wave calculation with the same
term from a molecular code.

### Challenge 4.3 — The cost of an Ewald sum and the optimal split

*Challenge · about 2 h · `chal4_3_ewald_cost.py`*

a) **Show** that ∑ᵢⱼ qᵢqⱼ cos(G·Rᵢⱼ) = |S(G)|² with S(G) = ∑ⱼ qⱼ exp(iG·Rⱼ).
   How does this change the cost of E_lr with the number of atoms?
b) Write a vectorized `ewald_fast` that uses this result. **Validate** it
   against your `ewald_energy` and against the Madelung constant of a
   2 × 2 × 2 NaCl supercell (64 ions).
c) Measure the wall time as a function of G_cut for CO in cells of 10 and 20
   bohr and for the NaCl supercell. Where is the optimum, and how does it
   depend on cell size and number of atoms? Compare with the recommendation
   G_cut = 2 bohr⁻¹ in [sec:est_ewald]. (In our runs, the optimum was
   about 1.5 bohr⁻¹ for CO in 10 bohr and for NaCl, and below 0.5 bohr⁻¹ for
   CO in 20 bohr; your timings will differ.)

---

## Section 5 — Self-consistent field procedure

The file `data/ch4_orbitals.npz` contains the five doubly occupied Kohn–Sham
orbitals of CH₄ (a0 = 10, N = 32, bare Coulomb nuclei, LDA with Slater
exchange and VWN5 correlation, all plane waves on the grid as basis) as
plane-wave coefficients, together with their orbital energies. It was produced
with the SCF program of Challenge 5.4.

### Exercise 5.1 — The total energy of methane from a set of orbitals

*Exercise · about 30 min · `ex5_1_total_energy.py` · write
`density_from_orbitals`, `energy_components`*

a) **Verify** that the orbitals are orthonormal and that the density
   [eq:edens] integrates to 10 electrons.
b) Calculate all terms of [eq:energy_total_discretized]. **Check:**

   | E_kin | E_nuc | E_H | E_xc | E_Ewald | E_total |
   |------:|------:|----:|-----:|--------:|--------:|
   | 31.907966 | −81.735803 | 19.038083 | −6.284507 | −0.481269 | −37.555531 |

c) Twice the sum of the orbital energies is −20.0103 Ht. **Show** how the
   total energy follows from it after correcting for the double-counted
   Hartree energy and for ν_xc versus ε_xc (cf. the text below
   [eq:edens_analytical]), and verify this numerically.
d) Estimate the largest kinetic energy ½|G|² that this grid can represent.
   Using the scaling E_cut ≈ 23 Z² Ht from Challenge 2.3, would you expect this
   total energy to be converged? Which electrons limit the convergence?

### Exercise 5.2 — Applying the Hamiltonian without storing it

*Exercise · about 40 min · `ex5_2_hamiltonian.py` · write
`effective_potential`, `apply_hamiltonian`, `hamiltonian_operator`,
`lowest_eigenstates_arpack`*

a) Implement [eq:appl_hamil]. **Verify** that H is Hermitian:
   ⟨x|Hy⟩ = ⟨Hx|y⟩ for random vectors x and y.
b) Construct ν_eff from the reference density. **Verify** that the reference
   orbitals are eigenvectors (residual ‖Hc − εc‖ ≈ 5 × 10⁻⁷), and recover the
   orbital energies with `scipy.sparse.linalg.eigsh` acting on a
   `LinearOperator`.
c) For a hydrogen atom on grids with N = 8 to 14, build the dense matrix by
   applying H to every unit vector, and compare the five lowest eigenvalues and
   the timings of `numpy.linalg.eigvalsh`, your `eigsh` routine and the
   provided block solver `lowest_eigenstates`. Extrapolate the dense timing to
   N = 32 using the N³_pw scaling. **Observe:** `eigsh` returns only two of the
   three degenerate 2p states and substitutes a higher state.
   **Interpret:** in exact arithmetic, a Krylov space built from a single start
   vector contains only one vector of each degenerate eigenspace. Why does a
   block method, which iterates one trial vector per state (cf. the Davidson
   method in Section 5), not suffer from this?

### Challenge 5.3 — Inside the Arnoldi method

*Challenge · about 2 h · `chal5_3_arnoldi.py`*

Implement a plain, unrestarted Arnoldi iteration (see [intermezzo:arnoldi])
with full re-orthogonalization, and run 300 steps on the CH₄ Hamiltonian of
Exercise 5.2 from a random start vector.

a) **Verify** that H_m is tridiagonal to rounding precision. Why?
b) **Verify** that the lowest Ritz value decreases monotonically and never
   drops below ε₁.
c) Plot the five lowest Ritz values against the step number, and the errors
   of the 1a₁ and 2a₁ levels on a logarithmic scale; compare with the figure in
   [intermezzo:arnoldi]. **Check:** the core level reaches an error of 10⁻⁶
   after about 30 steps, the valence level after about 105. Explain the
   difference using the spread of the spectrum (about 160 Ht here).
d) *(Optional)* How does the spread, and hence the number of steps, change
   with N? Relate this to the preconditioning in the provided block solver.

### Challenge 5.4 — Your own self-consistent field program

*Challenge · about 3 h · `chal5_4_scf.py`*

With everything in place, the SCF cycle of [fig:scf] takes about thirty lines.

a) Implement the cycle with a homogeneous initial density, linear mixing
   [sec:mixing] and a convergence test on both the energy and the
   root-mean-square density change [sec:scf_cycle]. Start each eigensolver
   call from the orbitals of the previous iteration. **Check** with H₂
   (bond length 1.4 bohr, a0 = 10, N = 32): E_total = −1.1373616 Ht.
b) Run CH₄. **Check:** your energy and orbital energies agree with the
   reference data of Exercise 5.1 to 10⁻⁶ Ht.
c) Count the SCF iterations as a function of β = 0.2, 0.4, …, 1.0 for CH₄
   (a0 = 10). For which β does linear mixing fail to converge? Section 5.3
   [sec:mixing] states that charge sloshing becomes worse in larger cells.
   **Test** this claim with H₂ in cells of 10, 15 and 20 bohr (N = 32, 48,
   64). Do you observe it? **Interpret:** which property of the system, rather
   than of the cell, determines how strongly a long-wavelength change of the
   density is amplified? *(Our runs: CH₄ needs 50, 23 and 15 iterations for
   β = 0.2, 0.4 and 0.6 and does not converge for β ≥ 0.8; H₂ needs essentially
   the same number of iterations in all three cells.)*
d) *(Optional)* Implement the Kerker preconditioner of [sec:mixing] and
   repeat (c).

### Exercise 5.5 — Real-valued orbitals and FFT upsampling

*Exercise · about 30 min · `ex5_5_real_orbitals.py`*

a) Using (Re z)² = (|z|² + Re z²)/2, **derive** that ∑(Re[ψe^{iθ}])² is
   maximal for θ = −½ arg ∑ψ². Apply this phase to every CH₄ orbital and
   compare the remaining imaginary fraction ∑(Im ψ)²/∑|ψ|² with that obtained
   by the recipe of [sec:orbitals] (phase at the grid point of largest |ψ|).
   **Observe:** 1a₁ and 2a₁ become real to machine precision, but some t₂
   orbitals keep about 10% imaginary weight. Why? (Read the last paragraph of
   [sec:orbitals] again.)
b) Upsample the real 2a₁ orbital by a factor of four [sec:orbitals],
   splitting the Nyquist coefficients. **Verify** that the original grid points
   are reproduced exactly and that the result is still real. The norm on the
   fine grid (0.999855) is slightly smaller than one: **explain** why, and
   predict its value from the coefficients.
c) Plot the orbital in the plane x = y (which contains C and two H atoms)
   before and after upsampling.

### Challenge 5.6 — A real basis for degenerate orbitals

*Challenge · about 1.5 h · `chal5_6_degenerate.py`*

a) **Argue** that, because the Hamiltonian is real, the real and imaginary
   parts of the three t₂ orbitals lie in their span. Use a singular value
   decomposition of these six real functions to construct three real
   orthonormal orbitals. **Check:** exactly three singular values are non-zero.
b) **Verify** that the new orbitals are orthonormal eigenfunctions with the
   t₂ orbital energy and give the same density.
c) *(Optional)* Rotate them onto functions resembling p_x, p_y and p_z centred
   on C (orthogonal Procrustes), and plot them. Compare with the familiar t₂
   orbitals of methane in a localized basis.

---

## Section 6 — Example calculations

From here on, complete calculations are carried out with PyPWDFT (version
0.3.0 or later), which implements everything you built in Section 5 and more:
a spherical cut-off with a separate density grid, Pulay mixing, a Davidson
eigensolver, the PBE functional and GTH pseudopotentials. The helper
`pwlib.diatomic(name, L)` builds H₂ or CO at the PBE bond lengths of
[sec:examples_setup], centred in a cubic cell of edge L. PyPWDFT stores grids
as `[z, y, x]`; `pwlib.from_pypwdft` converts them to the convention of pwlib.

### Exercise 6.1 — A complete calculation with PyPWDFT

*Exercise · about 25 min · `ex6_1_first_calculation.py`*

a) Run an all-electron PBE calculation of H₂ in a cell of 15 bohr at
   E_cut = 20 Ht. Inspect the energy components and the basis information.
   **Check:** the total energy lies 15 mHt above the reference value of
   [sec:examples_setup], as stated in [sec:examples_cutoff].
b) **Derive** an estimate of the number of plane waves from the volume of the
   sphere ½|G|² ≤ E_cut and the volume (2π)³/Ω that each G-vector occupies in
   reciprocal space. **Check:** your estimate agrees with PyPWDFT to within
   1% at 20, 40 and 80 Ht. Why is the density grid so much larger than
   N_pw^{1/3}? (See [sec:dualgrid].)
c) Repeat at 40 and 80 Ht and tabulate the error of the total energy
   (expected: 15.4, 6.1 and 2.2 mHt). Plot the occupied orbital; PyPWDFT has
   made it real-valued as described in [sec:orbitals].

### Exercise 6.2 — The cell size, absolute orbital energies and box states

*Exercise · about 30 min · `ex6_2_cell_size.py`*

Run H₂ at E_cut = 40 Ht in cells of 8, 10, 12, 15 and 20 bohr, with four
orbitals (one occupied, three unoccupied).

a) **Check:** the total energy at L = 8 bohr lies about 11 mHt below the
   converged value, as in [fig:ae_cell] (our run: 11.7 mHt), and is converged
   to 0.2 mHt from L = 12 bohr on. [sec:examples_cell] offers a *plausible
   explanation*: the tails of the orbital overlap with those of its periodic
   copies. **Test** it: how does the largest density on the faces of the cell
   change with L, compared with the energy error?
b) Between L = 12 and 20 bohr, by how much does the HOMO energy change, and by
   how much the total energy? (Our run: 3.2 mHt versus 0.2 mHt.) Explain the
   difference using [sec:examples_cell].
c) The unoccupied states are box states [sec:examples_cell]. Plot all orbital
   energies against 1/L² and print ε·L² for the unoccupied states. For which
   states is ε approximately proportional to 1/L², and what does that tell you
   about them? Why should you never interpret such a state as an
   antibonding σ* orbital?

### Challenge 6.3 — The price of a nucleus: all-electron CO

*Challenge · about 1.5 h (calculations: about 5 min) · `chal6_3_allelectron_co.py`*

a) Compute the total energy of H₂ (L = 15 bohr; 20 to 160 Ht) and of CO
   (L = 10 bohr; 50, 100 and 200 Ht) and plot the error with respect to the
   references of [sec:examples_setup] on log–log axes. **Check** against
   [sec:examples_cutoff]: H₂ is below 1 mHt at 160 Ht; CO is almost 24 Ht too
   high at 50 Ht, where the O 1s level lies at −12.5 Ht.
b) Tabulate the CO valence levels relative to the HOMO (5σ). At which cut-off
   do 4σ and 1π appear in the right order?
c) Fit error ≈ A·E_cut^{−p} for both molecules and extrapolate the cut-off
   for an error of 1 mHt. For CO, the estimate exceeds the one in
   [sec:examples_cutoff], which uses cut-offs up to 800 Ht (our fit over
   50–200 Ht: p = 0.82 and about 10⁷ Ht; H₂: p = 1.43 and about 140 Ht). Why
   does the extrapolation depend so strongly on the fit range? Relate the
   difference between the molecules to the Z² scaling of Challenge 2.3.

---

## Section 7 — Pseudopotentials

### Exercise 7.1 — The local part of a pseudopotential

*Exercise · about 35 min · `ex7_1_gth_local.py` · write `gth_local_g`,
`gth_external_potential`*

The parameters of the GTH-PBE pseudopotential of oxygen ([fig:gth_parameters])
are available as `pwlib.GTH_O`, and the real-space form [eq:gth_local] as
`pwlib.gth_local_r`.

a) Implement the reciprocal-space form [eq:gth_local_g], including the
   constant α at G = 0. **Check:** α = 0.134 Ht a.u.³ ([app:gth]) and
   ν_loc(0) ≈ −36 Ht ([sec:pp_atom]).
b) Plot −ν̃_loc(G) against ½|G|² together with 4πZ_ion/|G|² and the
   all-electron 4πZ/|G|² (Z = 8), as in [fig:pseudo_atom]c. At which kinetic
   energies does ν̃_loc fall below 50%, 10% and 1% of the ion's Coulomb tail?
   (Our values: 44, 89 and 141 Ht.) **Verify** your formula against a
   numerical radial Fourier transform of the short-range part
   ν_loc(r) + Z_ion/r.
c) Build ν_ext [eq:vext_pseudo] for one oxygen atom at the centre of cubic
   cells of 10, 15 and 20 bohr. At the nucleus, ν_ext differs from ν_loc(0).
   **Predict** this offset using what you learned in Sections 3 and 4, and
   verify your prediction. (Hint: the periodic images, the neutralizing
   background and the G = 0 coefficient α all contribute; the result is
   ξZ_ion/a0 + α/Ω to within 10⁻³ Ht.)

### Exercise 7.2 — A projector that only sees s electrons

*Exercise · about 30 min · `ex7_2_projector.py` · write `gth_projector_s_g`*

a) Implement the plane-wave coefficients of the s projector of oxygen
   ([app:gth], projectors). **Verify** that it is normalized and that its
   inverse transform reproduces [eq:gth_projector] times Y₀₀ = 1/√(4π).
b) Place the projector on O and compute ⟨p|ψ⟩ and h|⟨p|ψ⟩|² for normalized
   test functions: an s- and two p-type Gaussians on O, and two p-type
   Gaussians on C (1.1352 Å away along z). **Explain** which overlaps vanish
   exactly and why the p_z function on carbon is seen ([sec:pp_nonlocal]).
c) **Verify** that the overlap computed in real space [eq:nonlocal_action]
   equals the dot product of the coefficient vectors.

### Challenge 7.3 — Carbon monoxide with pseudopotentials

*Challenge · about 2 h (calculations: a few minutes) · `chal7_3_co_pseudo.py`*

a) Compute CO with GTH-PBE pseudopotentials (L = 10 bohr) at 20 to 160 Ht.
   **Check** against [sec:pp_co]: about −21.69 Ht and 63 000 plane waves at
   120 Ht; an error of 0.7 Ht at 20 Ht and about 1 mHt at 120 Ht, measured
   against the value at 160 Ht. Compare the number of plane waves needed for
   1 mHt with Challenge 6.3.
b) Tabulate the valence levels relative to the HOMO. Unlike the all-electron
   calculation, the order is correct already at 20 Ht. Why?
c) Using your projectors from Exercise 7.2 (and the carbon parameters in the
   skeleton), compute ⟨p_C|ψ⟩ and ⟨p_O|ψ⟩ for every orbital of the 80 Ht
   calculation, and reconstruct the non-local energy
   E_nl = 2∑ᵢ∑ₐ h_a|⟨p_a|ψᵢ⟩|² ([sec:pp_planewaves]). **Check:** your E_nl
   equals PyPWDFT's value to 10⁻⁸ Ht. Which orbitals are invisible to both
   projectors, and which has the most s character on oxygen? Compare with
   [sec:pp_co].
d) *(Optional, about 10 min of computing)* Determine the equilibrium bond
   length from a quadratic fit through five bond lengths between 1.125 and
   1.145 Å (L = 12 bohr, 120 Ht). Compare with [sec:pp_co]. (Our run:
   1.1358 Å, against 1.1356–1.1357 Å in the reader; the five-point quadratic
   fit is the likely cause of the small difference, but this was not checked.)
