# -*- coding: utf-8 -*-
"""
Plotting helpers shared by the Casimir scripts: lattice points, continuum curves, axis styles.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from modules.continuum import casimir_continuum
plt.rcParams.update({
    'text.usetex': True
})

PLOTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')#all the figures go here
os.makedirs(PLOTS, exist_ok=True)

NAMES = {'per': 'periodic', 'anti': 'antiperiodic', 'dir': 'Dirichlet', 'neu': 'Neumann', 'zar': 'Zaremba'}#used in file names
LETTER = {'per': 'P', 'anti': 'A', 'dir': 'D', 'neu': 'N', 'zar': 'Z'}#used in E_U labels
LABELS = {'per': 'Periodic', 'anti': 'Anti-periodic', 'dir': 'Dirichlet', 'neu': 'Neumann', 'zar': 'Zaremba'}#used in legends and titles
# Dirichlet and Neumann have exactly the same lattice Casimir energy (they differ only in the boundary energy C1),
# so the Casimir figures show 'dir' labelled as both
PLOT_BOUNDS = ['per', 'anti', 'dir', 'zar']
LABELS_DN = dict(LABELS, dir='Dirichlet = Neumann')
COLORS = {'per': 'C0', 'anti': 'C1', 'dir': 'C2', 'zar': 'C4'}

# default parameters of the Casimir figures
MASSES = [0.1, 0.2, 0.3, 0.4]
LV = [range(8, 25)] * len(MASSES)#N_L values for each mass
NA = {1: 96, 2: 96, 3: 96, 4: 100}#size of the periodic directions (time included)


def flat(v):
    """Fuse a nested list (one list per mass) into one array."""
    return np.array([a for row in v for a in row], dtype=float)


def plot_continuum(xmin, xmax, D, bound, label='Continuum', num=100, color=None):
    DP = np.linspace(xmin, xmax, num=num)
    plt.plot(DP, [casimir_continuum(mL, D, bound) for mL in DP], label=label, color=color)


def symlog(linthresh, ticks):
    plt.yscale('symlog', linthresh=linthresh)
    plt.yticks(ticks)


def save(fname, show=True):
    """Save the current figure in the plots folder."""
    plt.savefig(os.path.join(PLOTS, fname), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_Casimir(C, x, bound, D):
    """Lattice points and continuum curve of one dimension D (C, x: one list per mass)."""
    xdata = flat(x)
    plt.plot(xdata, flat(C), 'o', markersize=4, label='$D=%g$ lattice' % D)
    plot_continuum(min(xdata), max(xdata), D, bound, label='$D=%g$ continuum' % D)
    plt.ylabel('$E_%s/(Am^D)$' % LETTER[bound], fontsize=14)
    plt.xlabel('$mL$', fontsize=14)


def style_D(bound):
    """y axis and legend of the figures with all the dimensions together (values of casimir_exact)."""
    if bound in ['anti', 'zar']:#positive Casimir energy
        plt.yscale('log')
    else:#negative Casimir energy, down to ~1e-13
        plt.yscale('symlog', linthresh=1e-13)
    plt.legend(prop={'size': 8}, ncol=2)
