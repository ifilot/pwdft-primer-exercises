# -*- coding: utf-8 -*-
"""
Challenge 7.3 -- Carbon monoxide with pseudopotentials             [Challenge]

Outline only; see the description for the questions to answer.
Requires PyPWDFT >= 0.3.0.
"""

import matplotlib.pyplot as plt
import numpy as np
from pypwdft import PWDFT

import pwlib

# GTH-PBE parameters (r_s, h_11^s) of the s projectors, from the files
# pypwdft/psp/pbe/C-q4 and O-q6
PROJ = {'C': (0.30257575, 9.62248665), 'O': (0.22095592, 18.33745811)}


def main():
    L = 10.0
    ecuts = [20, 40, 60, 80, 120, 160]
    # (a) E_total versus E_cut, e.g.
    #       res = PWDFT(pwlib.diatomic('CO', L), cutoff=ecut, xc='pbe',
    #                   pseudopotential='gth').run(bands=7)
    #     compare with the primer: about -21.69 Ht and 63 000 plane waves at
    #     120 Ht, 0.7 Ht error at 20 Ht
    # (b) valence levels relative to the HOMO; compare with Challenge 6.3
    # (c) at 80 Ht: overlaps of your s projectors (pwlib.gth_projector_s_g)
    #     with every orbital, and E_nl reconstructed from them. Remember that
    #     PyPWDFT stores grids as [z, y, x]: use pwlib.from_pypwdft.
    # (d) optional: equilibrium bond length (cell of 12 bohr, 120 Ht)
    plt.show()


if __name__ == '__main__':
    main()
