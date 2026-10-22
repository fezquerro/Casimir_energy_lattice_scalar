# -*- coding: utf-8 -*-
"""
Continuum Casimir energy per unit area of a massive scalar field, E_U/(A m^D), as a function of mL.
D is the number of dimensions parallel to the plates (time included).
Bessel series: sum_c (+-1)^(c+1) K_{(D+1)/2}(a c mL)/c^{(D+1)/2}, summed until the relative size of a term is below ep
(scipy.special.kv, all the terms at once).
"""
import numpy as np
from scipy.special import kv, gamma, zeta


def _bessel_sum(z, nu, ep, alternating):
    if z <= 0:
        raise ValueError('mL must be > 0: the Bessel series diverges at mL=0 (use the massless formula)')
    n = 64
    while True:#double the number of terms until the last one is below ep relative to the sum
        c = np.arange(1, n + 1)
        t = kv(nu, z * c) / c**nu
        if alternating:
            t = t * (-1.0)**(c + 1)
        s = np.sum(t)
        if abs(t[-1]) <= ep * abs(s) or n >= 2**24:
            return s
        n *= 2


def Cas_Pergen(mL, D, ep=1e-12):
    auxD = (D + 1) / 2
    s = _bessel_sum(mL, auxD, ep, False)
    return -s / (pow(2, auxD - 1) * pow(np.pi, auxD) * pow(mL, auxD - 1))


def Cas_Dirgen(mL, D, ep=1e-12):
    """Dirichlet; Neumann has the same continuum Casimir energy."""
    auxD = (D + 1) / 2
    s = _bessel_sum(2 * mL, auxD, ep, False)
    return -s / (pow(2, D) * pow(np.pi, auxD) * pow(mL, auxD - 1))


def Cas_AntiPergen(mL, D, ep=1e-12):
    auxD = (D + 1) / 2
    s = _bessel_sum(mL, auxD, ep, True)
    return s / (pow(2, auxD - 1) * pow(np.pi, auxD) * pow(mL, auxD - 1))


def Cas_Zarembagen(mL, D, ep=1e-12):
    auxD = (D + 1) / 2
    s = _bessel_sum(2 * mL, auxD, ep, True)
    return s / (pow(2, D) * pow(np.pi, auxD) * pow(mL, auxD - 1))


_CONT = {'per': Cas_Pergen, 'anti': Cas_AntiPergen, 'dir': Cas_Dirgen, 'neu': Cas_Dirgen, 'zar': Cas_Zarembagen}


def casimir_continuum(mL, D, bound, ep=1e-12):
    """E_U/(A m^D) in the continuum for boundary condition bound ('per', 'anti', 'dir', 'neu', 'zar')."""
    return float(_CONT[bound](mL, D, ep))


def casimir_continuum_massless(D, bound):
    """
    Massless continuum Casimir energy E_U/A = c/L^D (thesis, Chapter 2); returns c.
    It is the limit mL -> 0 of (mL)^D casimir_continuum. With nu=(D+1)/2 and c_P = Gamma(nu) zeta(D+1)/pi^nu:
      periodic -c_P,  anti-periodic (1-2^-D) c_P,  Dirichlet = Neumann -c_P/2^(D+1),  Zaremba (2^D-1) c_P/2^(2D+1)
    e.g. D=3: -pi^2/90, 7pi^2/720, -pi^2/1440, 7pi^2/11520.
    """
    nu = (D + 1) / 2
    cP = gamma(nu) * zeta(D + 1) / np.pi**nu
    if bound == 'per':
        return -cP
    if bound == 'anti':
        return (1 - 2.0**-D) * cP
    if bound in ['dir', 'neu']:
        return -cP / 2**(D + 1)
    if bound == 'zar':
        return (2**D - 1) * cP / 2**(2 * D + 1)
    raise ValueError(f"Unknown boundary condition: {bound}")
