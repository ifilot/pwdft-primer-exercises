# Exercises: plane-wave density functional theory

Python exercises accompanying the primer *Plane-wave density functional
theory*. Over the course of the exercises you build a small plane-wave DFT
code yourself, `pwlib`, until it contains every ingredient of a
self-consistent field calculation; Challenge 5.4 assembles them into one.

The problem statements are in the exercise boxes at the end of Sections 2–7
of the primer.

## Two kinds of exercise

* **Exercise**: short (20–40 min). A skeleton script contains the checks; you
  write a few functions in `exercises/pwlib.py` and fill in short TODO blocks.
* **Challenge**: longer (1.5–3 h). The script contains an outline only.

## Getting started

```bash
pip install numpy scipy matplotlib
pip install threadpoolctl            # optional, see "A note on threads"
pip install "pypwdft>=0.3.0"         # only for Section 6 and Challenge 7.3
cd exercises
python ex2_1_grids.py
```

Each script prints `PASS` or `FAIL` for every check. An error that mentions
`None` or `NotImplementedError` means that a TODO block or a pwlib function is
still waiting for you. If you skip an exercise, copy the corresponding
functions from `solutions/pwlib.py` so that the later exercises still run.

Sections 2–5 and Exercises 7.1 and 7.2 use only NumPy, SciPy, Matplotlib and
your own `pwlib`. Section 6 and Challenge 7.3 use
[PyPWDFT](https://ifilot.github.io/pypwdft/) for complete calculations. Note
that PyPWDFT stores grids as `[z, y, x]`, whereas `pwlib` uses `[x, y, z]`;
`pwlib.from_pypwdft` converts between the two.

The reference solutions write their figures to `output/` next to the script.

### A note on threads

NumPy may start one thread per CPU core. For the small grids used here, that
mainly causes the threads to compete with each other and can make the
eigensolvers many times slower. If `threadpoolctl` is installed, `pwlib`
therefore limits the number of threads to 4. To change that, set the
environment variable `PWLIB_THREADS`.

## Layout

| Folder | Contents |
|--------|----------|
| `exercises/` | skeleton scripts and the student version of `pwlib.py` |
| `solutions/` | reference solutions and the complete `pwlib.py` |
| `data/` | CO 1π orbitals (HF/STO-3G) and CH₄ Kohn–Sham orbitals |
| `generators/` | scripts that produced the data and `exercises/pwlib.py` (not needed by students) |

## Testing the solutions

`bash run_solutions.sh` runs all reference solutions (about 15 minutes) and
reports every check that fails. `bash run_solutions.sh --quick` skips the five
scripts with complete SCF calculations (Exercises 6.1 and 6.2, Challenges 5.4,
6.3 and 7.3) and takes about a minute; this is what the GitHub workflow runs.
Run the full set locally before a release.
