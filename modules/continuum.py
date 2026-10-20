# -*- coding: utf-8 -*-
"""
Continuum Casimir energy per unit area of a massive scalar field, E_U/(A m^D), as a function of mL.
D is the number of dimensions parallel to the plates (time included).
Bessel series: sum_c (+-1)^(c+1) K_{(D+1)/2}(a c mL)/c^{(D+1)/2}, summed until the relative size of a term is below ep.
"""
import mpmath


def _bessel_sum(z, nu, ep, alternating):
    if z <= 0:
        raise ValueError('mL must be > 0: the Bessel series diverges at mL=0 (use the massless formula)')
    s = mpmath.besselk(nu, z)
    aux2 = s
    c = 1
    while abs(aux2 / s) > ep:
        c = c + 1
        aux2 = mpmath.besselk(nu, z * c) / pow(c, nu)
        s = s + (-aux2 if alternating and c % 2 == 0 else aux2)
    return s


def Cas_Pergen(mL, D, ep=1e-12):
    auxD = (D + 1) / 2
    s = _bessel_sum(mL, auxD, ep, False)
    return -s / (pow(2, auxD - 1) * pow(mpmath.pi, auxD) * pow(mL, auxD - 1))


def Cas_Dirgen(mL, D, ep=1e-12):
    """Dirichlet; Neumann has the same continuum Casimir energy."""
    auxD = (D + 1) / 2
    s = _bessel_sum(2 * mL, auxD, ep, False)
    return -s / (pow(2, D) * pow(mpmath.pi, auxD) * pow(mL, auxD - 1))


def Cas_AntiPergen(mL, D, ep=1e-12):
    auxD = (D + 1) / 2
    s = _bessel_sum(mL, auxD, ep, True)
    return s / (pow(2, auxD - 1) * pow(mpmath.pi, auxD) * pow(mL, auxD - 1))


def Cas_Zarembagen(mL, D, ep=1e-12):
    auxD = (D + 1) / 2
    s = _bessel_sum(2 * mL, auxD, ep, True)
    return s / (pow(2, D) * pow(mpmath.pi, auxD) * pow(mL, auxD - 1))


_CONT = {'per': Cas_Pergen, 'anti': Cas_AntiPergen, 'dir': Cas_Dirgen, 'neu': Cas_Dirgen, 'zar': Cas_Zarembagen}


def casimir_continuum(mL, D, bound, ep=1e-12):
    """E_U/(A m^D) in the continuum for boundary condition bound ('per', 'anti', 'dir', 'neu', 'zar')."""
    return float(_CONT[bound](mL, D, ep))
