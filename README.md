# Exercises: plane-wave density functional theory

Python exercises accompanying the primer *Plane-wave density functional
theory*. Over the course of the exercises you build a small plane-wave DFT
code yourself, ending with a self-consistent field calculation of methane.

The problem statements are in the exercise boxes at the end of Sections 2–7
of the primer.

## Two kinds of exercise

* **Exercise**: short (20–40 min). A skeleton script contains the checks; you
  write a few functions in `exercises/pwlib.py` and fill in short TODO blocks.
* **Challenge**: longer (1.5–4 h). The script contains an outline only.

## Getting started

```bash
pip install numpy scipy matplotlib threadpoolctl
pip install "pypwdft>=0.3.0"     # only for Sections 6 and 7
cd exercises
python ex2_1_grids.py
```

Each script prints `PASS` or `FAIL` for every check. An error that mentions
`None` or `NotImplementedError` means that a TODO block or a pwlib function is
still waiting for you. If you skip an exercise, copy the corresponding
functions from `solutions/pwlib.py` so that the later exercises still run.

Figures are written to `output/` next to the script and shown on screen.

Sections 2–5 use only NumPy, SciPy and your own `pwlib`. Sections 6 and 7
use [PyPWDFT](https://pypwdft.imc-tue.nl/) for complete calculations. Note
that PyPWDFT stores grids as `[z, y, x]`, whereas `pwlib` uses `[x, y, z]`;
`pwlib.from_pypwdft` converts between the two.

### A note on threads

NumPy may start one thread per CPU core. For the small grids used here, that
mainly causes the threads to compete with each other and can make the
eigensolvers many times slower. `pwlib` therefore limits the number of threads
to 4. To change that, set the environment variable `PWLIB_THREADS`.

## Layout

| Folder | Contents |
|--------|----------|
| `exercises/` | skeleton scripts and the student version of `pwlib.py` |
| `solutions/` | reference solutions and the complete `pwlib.py` |
| `data/` | CO 1π orbitals (HF/STO-3G) and CH₄ Kohn–Sham orbitals |
| `generators/` | scripts that produced the data and `exercises/pwlib.py` (not needed by students) |