# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 17:27:39 2026

@author: fer

Lattice energy and Casimir energy of the free scalar field.
L_periodic = [Lt, Lx, ...] are the periodic directions (the first one is time),
Ly (or NL) is the size of the direction with boundary condition bound:
'per' periodic, 'anti' anti-periodic, 'dir' Dirichlet, 'neu' Neumann, 'zar' Zaremba.
"""
import numpy as np

BOUNDS = ['per', 'anti', 'dir', 'neu', 'zar']


def phys_length(NL, bound):
    """Physical length of the confined direction in lattice units: N_L-1/2 for Zaremba, N_L otherwise."""
    return NL - 0.5 if bound == 'zar' else NL


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
    Energy per site, summing explicitly over all the modes.
    D = len(L_periodic) periodic directions (the first one is time) plus the direction Ly with boundary condition bound.
    For D=2, call as ener_Df([Lt, Lx], Ly, m, bound).
    """
    D = len(L_periodic)
    lambda_y = _confined_modes(Ly, bound)
    volume = np.prod(L_periodic) * Ly

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

def _transverse_modes(L_periodic, m):
    """S = sum_mu sin^2(pi n_mu/N_mu) + m^2/4 and lambda_t on the grid of the periodic directions."""
    D = len(L_periodic)
    S = np.full(tuple(L_periodic), m * m / 4, dtype=float)
    for i, L in enumerate(L_periodic):
        resh_shape = [1] * D
        resh_shape[i] = L
        S += ((np.sin(np.pi * np.arange(L) / L))**2).reshape(resh_shape)
    resh_t = [L_periodic[0] if j == 0 else 1 for j in range(D)]
    lt = np.broadcast_to(((np.sin(np.pi * np.arange(L_periodic[0]) / L_periodic[0]))**2).reshape(resh_t), S.shape)
    # S=0 only for m=0 and all n_mu=0, where lambda_t=0: that term never contributes, so it is dropped
    keep = S > 0
    return S[keep], lt[keep]


def _P(K, th, shch):
    """Periodic chain of length K: sum_k 1/(sin^2(pi k/K)+S) = K coth(K theta)/(sinh theta cosh theta)."""
    return K / np.tanh(K * th) / shch


def F_closed(N, S, bound):
    """sum over the modes of the confined direction of 1/(S+lambda_y), for every boundary condition."""
    th = np.arcsinh(np.sqrt(S))
    shch = np.sqrt(S * (1 + S))
    if bound == 'per':
        return _P(N, th, shch)
    if bound == 'anti':
        return _P(2 * N, th, shch) - _P(N, th, shch)
    if bound == 'dir':
        return (_P(2 * N, th, shch) - 1 / S - 1 / (1 + S)) / 2
    if bound == 'neu':
        return (_P(2 * N, th, shch) + 1 / S - 1 / (1 + S)) / 2
    if bound == 'zar':
        return (_P(4 * N - 2, th, shch) - _P(2 * N - 1, th, shch) - 1 / (1 + S)) / 2
    raise ValueError(f"Unknown boundary condition: {bound}")


def _ener_modes(S, lt, norm, Ly, m, bound):
    """ener_fast with the transverse modes already computed (norm = prod(L_periodic))."""
    volume = norm * Ly
    return _infrared(Ly, volume, m, bound) - np.sum(lt * F_closed(Ly, S, bound)) / volume


def ener_fast(L_periodic, Ly, m, bound):
    """
    Same result as ener_Df (energy per site), with the sum over the Ly direction done analytically.
    Cost: one sum over prod(L_periodic) modes instead of prod(L_periodic)*Ly.
    """
    S, lt = _transverse_modes(L_periodic, m)
    return _ener_modes(S, lt, np.prod(L_periodic), Ly, m, bound)


def ener_periodic(L_periodic, m):
    """Energy per site of the lattice made only of the periodic directions L_periodic (no confined direction)."""
    S, lt = _transverse_modes(L_periodic, m)
    norm = np.prod(L_periodic)
    return _infrared(norm, norm, m, 'per') - np.sum(lt / S) / norm


def bulk_C0(L_periodic, m):
    """Exact bulk energy density C0 = 1/2 - <lambda_t/sqrt(S(1+S))> (limit N_L0 -> infinity of thesis eq. 3.78)."""
    S, lt = _transverse_modes(L_periodic, m)
    return 0.5 - np.sum(lt / np.sqrt(S * (1 + S))) / np.prod(L_periodic)


def boundary_C1(L_periodic, m, bound):
    """
    Exact boundary energy per unit area C1^U.
    Equivalently C1^D = -(e0+e1)/2, C1^N = (e0-e1)/2, C1^Z = -e1/2 with e0 = bulk energy one dimension
    less at mass m and e1 the same at mass sqrt(m^2+4).
    For m=0, 'per' and 'neu' include the term -1/(2 N0 A) from the removed constant mode,
    so that N_L*ener_Df = bulk_C0*L_U + boundary_C1 + casimir_exact holds exactly also at m=0.
    """
    norm = np.prod(L_periodic)
    zero_mode = -0.5 / norm if (m == 0 and bound in ['per', 'neu']) else 0.0
    if bound == 'anti':
        return 0.0
    if bound == 'per':
        return zero_mode
    S, lt = _transverse_modes(L_periodic, m)
    if bound == 'dir':
        return -0.5 + 0.5 * np.sum(lt * (1 + 2 * S) / (S * (1 + S))) / norm
    if bound == 'neu':
        return -0.5 * np.sum(lt / (S * (1 + S))) / norm + zero_mode
    if bound == 'zar':
        return -0.25 + 0.5 * np.sum(lt / (1 + S)) / norm
    raise ValueError(f"Unknown boundary condition: {bound}")


def casimir_exact(L_periodic, NL, m, bound):
    """
    Casimir energy per unit transverse area, E_U/A, in the limit N_L0 -> infinity.
    No reference lattice is needed. Divide by m**D (D=len(L_periodic)) to compare with the continuum.
    NL is the size of the direction with boundary condition bound (an int, or a list to reuse the transverse modes).
    For 'zar' the physical length is NL-1/2.
    """
    S, lt = _transverse_modes(L_periodic, m)
    th = np.arcsinh(np.sqrt(S))
    w = lt / np.sqrt(S * (1 + S))
    norm = np.prod(L_periodic)

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
        return -np.sum(w * g) / norm
    return [one(N) for N in NL] if np.ndim(NL) else one(NL)


def casimir_finite_ref(L_periodic, NL, NL0, m, bound):
    """
    Thesis prescription (3.79)-(3.87) with a finite reference lattice NL0, evaluated with the closed-form sum.
    casimir_exact is its NL0 -> infinity limit. NL can be a list, as in casimir_exact.
    """
    S, lt = _transverse_modes(L_periodic, m)
    norm = np.prod(L_periodic)
    e_per = _ener_modes(S, lt, norm, NL0, m, 'per')#bulk reference
    if bound in ['per', 'anti']:
        def one(N):
            return (_ener_modes(S, lt, norm, N, m, bound) - e_per) * N
    else:
        e_ref = _ener_modes(S, lt, norm, NL0, m, bound)#same boundaries, length NL0

        def one(N):
            return _ener_modes(S, lt, norm, N, m, bound) * N - e_ref * NL0 - (N - NL0) * e_per
    return [one(N) for N in NL] if np.ndim(NL) else one(NL)


def casimir_second_diff(L_periodic, NL, L0, m, bound):
    """
    Second-difference renormalization: E(N_L) + E(N_L+2L0) - 2E(N_L+L0), with E = N * energy per site.
    The bulk and boundary terms cancel, leaving the Casimir energy for large L0.
    """
    S, lt = _transverse_modes(L_periodic, m)
    norm = np.prod(L_periodic)

    def E(N):
        return _ener_modes(S, lt, norm, N, m, bound) * N
    return E(NL) + E(NL + 2 * L0) - 2 * E(NL + L0)


def casimir_scan(L_periodic, Lv, m, bound, NL0=None):
    """
    E_U/(A m^D) and the physical length mL for every mass m[i] and every N_L in Lv[i].
    NL0 is the size of the reference lattice; None gives the exact N_L0 -> infinity result.
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
