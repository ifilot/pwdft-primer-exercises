# -*- coding: utf-8 -*-
"""
Challenge 6.3 -- The price of a nucleus: all-electron CO           [Challenge]

Outline only; see the description for the questions to answer.
Requires PyPWDFT >= 0.3.0. The CO calculation at 200 Ht takes a few minutes.
"""

import matplotlib.pyplot as plt
import numpy as np
from pypwdft import PWDFT

import pwlib

E_REF = {'H2': -1.16673, 'CO': -113.2424}     # primer, Section 6
CELL = {'H2': 15.0, 'CO': 10.0}
ECUTS = {'H2': [20, 40, 80, 160], 'CO': [50, 100, 200]}


def main():
    # (a) total energies of H2 and CO versus E_cut, e.g.
    #       res = PWDFT(pwlib.diatomic('CO', 10.0), cutoff=50, xc='pbe',
    #                   pseudopotential='all-electron').run(bands=9)
    #     compare with the primer: CO error at 50 Ht, O 1s level at 50 Ht
    # (b) valence levels of CO relative to the HOMO; in which order do they
    #     appear at each cut-off?
    # (c) fit error = A E_cut^-p and extrapolate to 1 mHt
    plt.show()


if __name__ == '__main__':
    main()
