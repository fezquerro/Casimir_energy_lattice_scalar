# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 17:27:39 2026

@author: fer

Lattice energy and Casimir energy of the free scalar field.
L_periodic = [Lt, Lx, ...] are the periodic directions (the first one is time),
Ly (or NL) is the size of the direction with boundary condition bound:
'per' periodic, 'anti' anti-periodic, 'dir' Dirichlet, 'neu' Neumann, 'zar' Zaremba.
"""
import math
from collections import Counter
import numpy as np
from scipy.integrate import quad
from scipy.special import ive

BOUNDS = ['per', 'anti', 'dir', 'neu', 'zar']


def phys_length(NL, bound):
    """Physical length of the confined direction in lattice units: N_L-1/2 for Zaremba, N_L otherwise."""
    return NL - 0.5 if bound == 'zar' else NL


def _size(L_periodic):
    """Number of sites of the periodic directions, as a float. The product is done with Python ints:
    np.prod gives int32 on Windows, which overflows above 2^31 (e.g. 224^4)."""
    return float(math.prod(int(L) for L in L_periodic))


def _confined_modes(Ly, bound):
    """Eigenvalues sin^2(...) of the confined direction (the lattice Laplacian ones divided by 4)."""
    if bound == 'dir':  # Dirichlet bc
        return (np.sin(np.pi * np.arange(1, Ly) / (Ly * 2)))**2
    if bound == 'per':  # Periodic bc
        return (np.sin(np.pi * np.arange(0, Ly) / Ly))**2
    if bound == 'anti':  # Anti-periodic bc
        return (np.sin(np.pi * (np.arange(0, Ly) + 0.5) / Ly))**2
    if bound == 'neu':  # Neumann bc
        return (np.sin(np.pi * np.arange(0, Ly) / (Ly * 2)))**2
    if bound == 'zar':  # Zaremba bc
        return (np.sin(np.pi * (np.arange(0, Ly - 1) + 0.5) / (2 * Ly - 1)))**2
    raise ValueError(f"Unknown boundary condition: {bound}")


def _infrared(Ly, volume, m, bound):
    """Infrared term of the energy per site; for m=0, 'per' and 'neu' lose the constant mode."""
    if bound in ['dir', 'zar']:
        return 0.5 * (1 - 1 / Ly)
    if m == 0 and bound in ['per', 'neu']:
        return 0.5 * (1 - 1 / volume)
    return 0.5


def ener_Df(L_periodic, Ly, m, bound):
    """
    Energy per site, summing explicitly over all the modes (reference implementation, used in Checks.py).
    D = len(L_periodic) periodic directions (the first one is time) plus the direction Ly with boundary condition bound.
    For D=2, call as ener_Df([Lt, Lx], Ly, m, bound).
    """
    D = len(L_periodic)
    lambda_y = _confined_modes(Ly, bound)
    volume = _size(L_periodic) * Ly

    # Build the D-dimensional array of the periodic directions using broadcasting
    S = np.full(tuple(L_periodic), m * m / 4, dtype=float)#add the mass term to the denom
    for i, L in enumerate(L_periodic):
        resh_shape = [1] * D
        resh_shape[i] = L#shape [1]...[Li]..[1]
        S += ((np.sin(np.pi * np.arange(L) / L))**2).reshape(resh_shape)#sum the eigenvalues of that direction

    # lambda_t reshaped to have full dim in first axis
    resh_t = [L_periodic[0] if j == 0 else 1 for j in range(D)]
    lambda_t_resh = ((np.sin(np.pi * np.arange(L_periodic[0]) / L_periodic[0]))**2).reshape(resh_t)

    # sum of lambda_t / denom over all modes, one ny mode at a time
    # (memory of a single D-dimensional array instead of the full (D+1)-dimensional one)
    aux = 0.0
    for k in range(len(lambda_y)):
        denom = S + lambda_y[k]
        # Handle zero mode for per or neu when m==0 (lambda_y[0]=0 for both)
        if (bound in ['per', 'neu']) and m == 0 and k == 0:
            denom[tuple(0 for _ in range(D))] = 1#lambda_t=0 there, so this term sums zero
        aux += np.sum(lambda_t_resh / denom)
    return _infrared(Ly, volume, m, bound) - aux / volume


# ----------------------------------------------------------------------------
# Closed-form sum over the confined direction
# Key identity: sum_{n=0}^{N-1} 1/(sin^2(pi n/N)+S) = N coth(N theta)/(sinh theta cosh theta), sinh^2 theta = S
# ----------------------------------------------------------------------------

def _distinct_sums(N, k):
    """
    Distinct values of sum_{i=1..k} sin^2(pi n_i/N) over k directions of size N, and how many (n_1..n_k) give each:
    sin^2(pi n/N) = sin^2(pi (N-n)/N), and the order of the k directions does not matter.
    """
    if k == 0:
        return np.zeros(1), np.ones(1)
    n = np.arange(N // 2 + 1)
    v = np.sin(np.pi * n / N)**2
    c = np.where((n == 0) | (2 * n == N), 1, 2)#n and N-n give the same value
    idx = np.indices((len(v),) * k).reshape(k, -1).T
    idx = idx[np.all(idx[:, 1:] >= idx[:, :-1], axis=1)]#one ordering of each set of k values
    rep = np.ones(len(idx))#product of the factorials of the repeated indices
    for j in range(k):
        rep *= np.sum(idx[:, :j + 1] == idx[:, [j]], axis=1)
    weight = math.factorial(k) / rep * np.prod(c[idx], axis=1)
    return v[idx].sum(axis=1), weight


class _Modes:
    """
    Transverse modes of the periodic directions, only the distinct ones, each with its multiplicity w:
    S = sum_mu sin^2(pi n_mu/N_mu) + m^2/4, lambda_t = sin^2(pi n_0/N_0), theta = arcsinh(sqrt(S)),
    shch = sinh(theta) cosh(theta) = sqrt(S(1+S)). A sum over all the modes of lambda_t*f is sum(wlt*f).
    """
    def __init__(self, L_periodic, m):
        self.norm = _size(L_periodic)
        lt0, w0 = _distinct_sums(L_periodic[0], 1)#time direction, apart because lambda_t enters the energy
        sums, w = np.zeros(1), np.ones(1)
        for N, k in Counter(L_periodic[1:]).items():#spatial directions, grouped by size
            s_k, w_k = _distinct_sums(N, k)
            sums = (sums[:, None] + s_k[None, :]).ravel()
            w = (w[:, None] * w_k[None, :]).ravel()
        S = (lt0[:, None] + sums[None, :]).ravel() + m * m / 4
        lt = np.repeat(lt0, len(sums))
        w = (w0[:, None] * w[None, :]).ravel()
        # S=0 only for m=0 and all n_mu=0, where lambda_t=0: that term never contributes, so it is dropped
        keep = S > 0
        self.S, self.wlt = S[keep], (w * lt)[keep]
        self.th = np.arcsinh(np.sqrt(self.S))
        self.shch = np.sqrt(self.S * (1 + self.S))


def _P(K, modes):
    """Periodic chain of length K: sum_k 1/(sin^2(pi k/K)+S) = K coth(K theta)/(sinh theta cosh theta)."""
    return K / np.tanh(K * modes.th) / modes.shch


def F_closed(N, modes, bound):
    """sum over the modes of the confined direction of 1/(S+lambda_y), for every boundary condition."""
    S = modes.S
    if bound == 'per':
        return _P(N, modes)
    if bound == 'anti':
        return _P(2 * N, modes) - _P(N, modes)
    if bound == 'dir':
        return (_P(2 * N, modes) - 1 / S - 1 / (1 + S)) / 2
    if bound == 'neu':
        return (_P(2 * N, modes) + 1 / S - 1 / (1 + S)) / 2
    if bound == 'zar':
        return (_P(4 * N - 2, modes) - _P(2 * N - 1, modes) - 1 / (1 + S)) / 2
    raise ValueError(f"Unknown boundary condition: {bound}")


def _ener_modes(modes, Ly, m, bound):
    """ener_fast with the transverse modes already computed."""
    volume = modes.norm * Ly
    return _infrared(Ly, volume, m, bound) - np.sum(modes.wlt * F_closed(Ly, modes, bound)) / volume


def ener_fast(L_periodic, Ly, m, bound):
    """
    Same result as ener_Df (energy per site), with the sum over the Ly direction done analytically
    and only the distinct transverse modes summed.
    """
    return _ener_modes(_Modes(L_periodic, m), Ly, m, bound)


def ener_periodic(L_periodic, m):
    """Energy per site of the lattice made only of the periodic directions L_periodic (no confined direction)."""
    modes = _Modes(L_periodic, m)
    return _infrared(modes.norm, modes.norm, m, 'per') - np.sum(modes.wlt / modes.S) / modes.norm


def bulk_C0(L_periodic, m):
    """Exact bulk energy density C0 = 1/2 - <lambda_t/sqrt(S(1+S))> (thesis eq. 3.78 estimates it with a reference lattice)."""
    modes = _Modes(L_periodic, m)
    return 0.5 - np.sum(modes.wlt / modes.shch) / modes.norm


def boundary_C1(L_periodic, m, bound):
    """
    Exact boundary energy per unit area C1^U.
    Equivalently C1^D = -(e0+e1)/2, C1^N = (e0-e1)/2, C1^Z = -e1/2 with e0 = bulk energy one dimension
    less at mass m and e1 the same at mass sqrt(m^2+4).
    For m=0, 'per' and 'neu' include the term -1/(2 N0 A) from the removed constant mode,
    so that N_L*ener_Df = bulk_C0*L_U + boundary_C1 + casimir_exact holds exactly also at m=0.
    """
    norm = _size(L_periodic)
    zero_mode = -0.5 / norm if (m == 0 and bound in ['per', 'neu']) else 0.0
    if bound == 'anti':
        return 0.0
    if bound == 'per':
        return zero_mode
    modes = _Modes(L_periodic, m)
    S, wlt = modes.S, modes.wlt
    if bound == 'dir':
        return -0.5 + 0.5 * np.sum(wlt * (1 + 2 * S) / (S * (1 + S))) / norm
    if bound == 'neu':
        return -0.5 * np.sum(wlt / (S * (1 + S))) / norm + zero_mode
    if bound == 'zar':
        return -0.25 + 0.5 * np.sum(wlt / (1 + S)) / norm
    raise ValueError(f"Unknown boundary condition: {bound}")


def casimir_exact(L_periodic, NL, m, bound):
    """
    Casimir energy per unit transverse area, E_U/A, computed directly from the closed-form sum.
    No reference lattice is needed (casimir_finite_ref tends to it for a large reference lattice). Divide by m**D (D=len(L_periodic)) to compare with the continuum.
    NL is the size of the direction with boundary condition bound (an int, or a list to reuse the transverse modes).
    For 'zar' the physical length is NL-1/2.
    """
    modes = _Modes(L_periodic, m)
    th = modes.th
    w = modes.wlt / modes.shch

    def one(N):
        if bound == 'per':
            x = 2 * N * th
            g = 2 * N * np.exp(-x) / (-np.expm1(-x))
        elif bound == 'anti':
            q = np.exp(-2 * N * th)
            g = -2 * N * q / (1 + q)
        elif bound in ['dir', 'neu']:
            x = 4 * N * th
            g = 2 * N * np.exp(-x) / (-np.expm1(-x))
        elif bound == 'zar':
            Le = N - 0.5
            q = np.exp(-4 * Le * th)
            g = -2 * Le * q / (1 + q)
        else:
            raise ValueError(f"Unknown boundary condition: {bound}")
        return -np.sum(w * g) / modes.norm
    return [one(N) for N in NL] if np.ndim(NL) else one(NL)


# ----------------------------------------------------------------------------
# Infinite transverse lattice (N_0, N_mu -> infinity: infinite walls at zero temperature)
# Every periodic direction is summed with (1/N) sum_n e^{-t sin^2(pi n/N)} = sum_w h_{|w|N}(t), h_k(t) = e^{-t/2} I_k(t/2),
# keeping only w=0 (no transverse winding); the time direction, weighted with lambda_0, gives kappa_0 = -dh_0/dt.
# All the functions hold also for m=0.
# ----------------------------------------------------------------------------

def _ive(k, x):
    """e^{-x} I_k(x). scipy's ive gives nan above x = 2^30; from x = 1e7 on the uniform (Debye) expansion
    with its first correction is used instead (relative error < 1e-11)."""
    if x < 1e7:
        return ive(k, x)
    sq = np.hypot(k, x)#sqrt(k^2+x^2) without overflow
    return np.exp(k * k / (sq + x) - k * np.arcsinh(k / x)) / np.sqrt(2 * np.pi * sq) * (1 + (3 - 5 * (k / sq)**2) / (24 * sq))


def _h(k, t):
    """h_k(t) = e^{-t/2} I_k(t/2): average over an infinite chain of e^{-t lambda} for walks that wind k sites."""
    return _ive(k, t / 2)


def _kappa0(t):
    """kappa_0(t) = -dh_0/dt = e^{-x}(I_0 - I_1)(x)/2, x = t/2: the same average with the weight lambda_0 of the time
    direction. The difference cancels at large x (loses log10(x) digits): for x >= 1e4 its asymptotic series
    (2 pi x)^{-1/2} [1/(2x) + 3/(16x^2) + 45/(256x^3) + 25200/(98304x^4) + O(x^-5)] is used."""
    x = t / 2
    if x < 1e4:
        return (_ive(0, x) - _ive(1, x)) / 2
    y = 1 / x
    return y * (0.5 + y * (3 / 16 + y * (45 / 256 + y * 25200 / 98304))) / np.sqrt(2 * np.pi * x) / 2


def _t_integral(f, lo, hi):
    """int_0^infinity f(t) dt in the variable u = ln t, int du e^u f(e^u): power-law tails of f (m=0) become
    exponential decays. The integrand changes between t = lo and hi, where the u range is split every decade.
    The absolute tolerance of each piece is relative to the sum of the previous ones (the tails can be negligible)."""
    def g(u):
        return np.exp(u) * f(np.exp(u)) if u < 700 else 0.0
    n = max(1, int(np.ceil(np.log10(hi / lo))))
    edges = [-np.inf] + list(np.log(lo) + np.log(10) * np.arange(n + 1)) + [np.inf]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        total += quad(g, a, b, limit=200, epsabs=1e-14 * abs(total), epsrel=1e-12)[0]
    return total


def _decay(m2):
    """Scale 4/m2 of e^{-t m2/4}; for m2 = 0 the integrands decay as a power of t, taken care of by _t_integral."""
    return 4 / m2 if m2 > 0 else 1e4


def _J(D, m, c):
    """J(c) = int_0^inf dt e^{-t(M^2/4+c)} kappa_0 h_0^{D-1}, the limit of <lambda_0/(S+c)> (article eq. Jc)."""
    return _t_integral(lambda t: np.exp(-t * (m * m / 4 + c)) * _kappa0(t) * _h(0, t)**(D - 1), 0.1, 10 * _decay(m * m + 4 * c))


def bulk_C0_inf(D, m):
    """Bulk energy density of the infinite transverse lattice with D periodic directions (article eq. C0inf):
    C0^inf = 1/2 - int_0^inf dt e^{-t M^2/4} kappa_0 h_0^D. bulk_C0 tends to it as N_A grows."""
    return 0.5 - _t_integral(lambda t: np.exp(-t * m * m / 4) * _kappa0(t) * _h(0, t)**D, 0.1, 10 * _decay(m * m))


def boundary_C1_inf(D, m, bound):
    """
    Boundary energy of the infinite transverse lattice: C1^P = C1^A = 0,
    C1^D = -1/2 + J(0)/2 + J(1)/2, C1^N = -J(0)/2 + J(1)/2, C1^Z = -1/4 + J(1)/2.
    boundary_C1 tends to it as N_A grows.
    """
    if bound in ['per', 'anti']:
        return 0.0
    J1 = _J(D, m, 1)
    if bound == 'zar':
        return -0.25 + 0.5 * J1
    J0 = _J(D, m, 0)
    if bound == 'dir':
        return -0.5 + 0.5 * J0 + 0.5 * J1
    if bound == 'neu':
        return -0.5 * J0 + 0.5 * J1
    raise ValueError(f"Unknown boundary condition: {bound}")


def casimir_inf(D, NL, m, bound):
    """
    Casimir energy per unit area of infinite walls at zero temperature, D periodic directions:
    E_U^inf/A = -2 L_U int_0^inf dt e^{-t M^2/4} kappa_0 h_0^{D-1} W(t), with the windings W = sum_{w>=1} sigma_U^w h_{w l_U}
    and (l_U, sigma_U) = (N_L, +1) P, (N_L, -1) A, (2N_L, +1) D and N, (2N_L-1, -1) Z.
    For t < l^2/4 W is summed directly (h_{wl} ~ e^{-(wl)^2/t}: four windings are exact to round-off); for larger t
    with the chain identity read backwards, W = [(1/l) sum_n e^{-t sin^2(pi(n+phi)/l)} - h_0]/2, phi = 0 (sigma=+1)
    or 1/2 (sigma=-1), which sums all the windings at once and also holds at m=0.
    casimir_exact tends to it as N_A grows. NL can be a list, as in casimir_exact.
    """
    ell, sigma = {'per': (1, 1), 'anti': (1, -1), 'dir': (2, 1), 'neu': (2, 1), 'zar': (2, -1)}[bound]

    def one(N):
        l = ell * int(N) - (1 if bound == 'zar' else 0)#Python int: no int32 overflow in l*l
        lam = np.sin(np.pi * (np.arange(l) + (0 if sigma > 0 else 0.5)) / l)**2

        def f(t):
            if t < l * l / 4:
                w = sigma * _h(l, t) + _h(2 * l, t) + sigma * _h(3 * l, t) + _h(4 * l, t)
            else:
                w = 0.5 * (np.mean(np.exp(-t * lam)) - _h(0, t))
            return np.exp(-t * m * m / 4) * _kappa0(t) * _h(0, t)**(D - 1) * w
        return -2 * phys_length(N, bound) * _t_integral(f, 0.1, 10 * max(l * l, _decay(m * m)))
    return [one(N) for N in NL] if np.ndim(NL) else one(NL)


def casimir_finite_ref(L_periodic, NL, NL0, m, bound):
    """
    Thesis prescription (3.79)-(3.87) with a finite reference lattice NL0, evaluated with the closed-form sum.
    casimir_exact is its NL0 -> infinity limit. NL can be a list, as in casimir_exact.
    """
    modes = _Modes(L_periodic, m)
    e_per = _ener_modes(modes, NL0, m, 'per')#bulk reference
    if bound in ['per', 'anti']:
        def one(N):
            return (_ener_modes(modes, N, m, bound) - e_per) * N
    else:
        e_ref = _ener_modes(modes, NL0, m, bound)#same boundaries, length NL0

        def one(N):
            return _ener_modes(modes, N, m, bound) * N - e_ref * NL0 - (N - NL0) * e_per
    return [one(N) for N in NL] if np.ndim(NL) else one(NL)


def casimir_second_diff(L_periodic, NL, L0, m, bound):
    """
    Second-difference renormalization: E(N_L) + E(N_L+2L0) - 2E(N_L+L0), with E = N * energy per site.
    The bulk and boundary terms cancel, leaving the Casimir energy for large L0.
    """
    modes = _Modes(L_periodic, m)

    def E(N):
        return _ener_modes(modes, N, m, bound) * N
    return E(NL) + E(NL + 2 * L0) - 2 * E(NL + L0)


def casimir_scan(L_periodic, Lv, m, bound, NL0=None):
    """
    E_U/(A m^D) and the physical length mL for every mass m[i] and every N_L in Lv[i].
    NL0 is the size of the reference lattice; None gives the exact result (casimir_exact).
    """
    D = len(L_periodic)
    C = []
    x = []
    for mi, NLs in zip(m, Lv):
        NLs = list(NLs)
        if NL0 is None:
            E = casimir_exact(L_periodic, NLs, mi, bound)
        else:
            E = casimir_finite_ref(L_periodic, NLs, NL0, mi, bound)
        C.append([e / mi**D for e in E])
        x.append([mi * phys_length(N, bound) for N in NLs])
    return C, x
