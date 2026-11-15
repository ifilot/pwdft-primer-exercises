# -*- coding: utf-8 -*-
"""
Challenge 3.3 -- Measuring the Madelung constant                       [Challenge]

Outline only; see the description for the questions to answer.
"""

import matplotlib.pyplot as plt
import numpy as np

import pwlib

XI = 2.837297                # literature value, for comparison only
SPACING = 10.0 / 32          # fixed grid spacing (bohr)


def electrostatics(a0, alpha):
    """TODO: E_rep and E_nuc of a normalized Gaussian cloud (exponent alpha)
    centred on a proton in a cubic cell of edge a0, at fixed grid spacing."""
    raise NotImplementedError


def fit(a0s, dE):
    """TODO: least-squares fit dE(a0) = p / a0 + q / a0^3; return (p, q).
    Hint: np.linalg.lstsq with the columns 1/a0 and 1/a0^3."""
    raise NotImplementedError


def main():
    a0s = np.array([10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0])
    # for alpha = 2 and alpha = 1:
    #   * compute the deviations from the isolated values,
    #       E_rep^iso = 2 sqrt(alpha / (2 pi)),  E_nuc^iso = -2 sqrt(alpha / pi)
    #   * fit them, and compare p with xi and q with the spread terms
    #   * plot |deviation| versus a0 on log-log axes, including the sum
    plt.show()


if __name__ == '__main__':
    main()
